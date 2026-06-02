"""
src/waves.py
------------
Parsers for GPSWaves output files:
  - waves.log    → wave bulk parameters (Hs, Ta, Tp, Dp) per 30-min burst
  - spectrum.log → full directional wave spectrum per 30-min burst

Both files share the same PERIOD header format (doc 06258, section 7):
  PERIOD, start_date, start_time, start_usec, start_lat, start_lon,
          end_date,   end_time,   end_usec,   end_lat,   end_lon,
          n_samples, sampling_freq, [6 config fields]

Public API
----------
  load_waves(data_path, file_pattern)   → pd.DataFrame  (one row per burst)
  load_spectra(data_path, file_pattern) → xr.Dataset     (time × frequency)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Sentinel value
# ---------------------------------------------------------------------------
INVALID = 9999.0


# ---------------------------------------------------------------------------
# PERIOD header parser
# ---------------------------------------------------------------------------

def _parse_period(parts: list[str]) -> dict:
    """
    Parse a PERIOD row into a dict with start_time, end_time, n_samples,
    sampling_freq, start_lat, start_lon, end_lat, end_lon.
    Returns None if the row cannot be parsed.
    """
    try:
        # PERIOD, date, time, usec, lat, lon, date, time, usec, lat, lon,
        #         n_samples, sampling_freq, [6 config fields]
        # parts[0] == 'PERIOD'
        start_dt  = pd.Timestamp(f"{parts[1]} {parts[2]}", tz='UTC')
        start_lat = float(parts[4])
        start_lon = float(parts[5])
        end_dt    = pd.Timestamp(f"{parts[6]} {parts[7]}", tz='UTC')
        end_lat   = float(parts[9])
        end_lon   = float(parts[10])
        n_samples = int(parts[11])
        fs        = float(parts[12])
        return dict(
            start_time   = start_dt,
            end_time     = end_dt,
            start_lat    = start_lat,
            start_lon    = start_lon,
            end_lat      = end_lat,
            end_lon      = end_lon,
            n_samples    = n_samples,
            sampling_freq= fs,
        )
    except Exception as e:
        warnings.warn(f"waves.py: could not parse PERIOD row: {e}")
        return None


def _parse_row(label: str, parts: list[str], n: int) -> Optional[np.ndarray]:
    """
    Parse a data row (e.g. ENERGYSPECTRA, GPSWAVES, …).
    Returns array of floats, or None if all values are 9999 (invalid burst).
    """
    vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    if len(vals) == 1 and vals[0] == INVALID:
        return None          # single 9999 sentinel
    if np.all(vals == INVALID):
        return None
    # Replace individual 9999 with NaN
    vals[vals == INVALID] = np.nan
    return vals


# ---------------------------------------------------------------------------
# waves.log parser
# ---------------------------------------------------------------------------

def load_waves(
    data_path:    Path,
    file_pattern: str = '*waves.log',
) -> pd.DataFrame:
    """
    Parse all waves.log files under data_path.

    Returns a DataFrame indexed by burst start_time with columns:
        Hs, Ta, Tp, Dp, n_samples, sampling_freq,
        start_lat, start_lon, end_lat, end_lon, end_time, valid
    """
    data_path = Path(data_path)
    files = sorted(data_path.rglob(file_pattern))
    if not files:
        warnings.warn(f"load_waves: no files matching '{file_pattern}' under {data_path}")
        return pd.DataFrame()

    rows = []
    for fpath in files:
        print(f"  parsing {fpath.name} …", end=' ')
        n_valid = n_invalid = 0

        period = None
        with open(fpath, 'r', errors='replace') as f:
            for line in f:
                line  = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]

                if parts[0] == 'PERIOD':
                    period = _parse_period(parts)

                elif parts[0] == 'GPSWAVES' and period is not None:
                    vals = _parse_row('GPSWAVES', parts, 4)
                    row  = dict(period)
                    if vals is None or len(vals) < 3:
                        row.update(Hs=np.nan, Ta=np.nan, Tp=np.nan,
                                   Dp=np.nan, valid=False)
                        n_invalid += 1
                    else:
                        row.update(
                            Hs    = float(vals[0]),
                            Ta    = float(vals[1]) if len(vals) > 1 else np.nan,
                            Tp    = float(vals[2]) if len(vals) > 2 else np.nan,
                            Dp    = float(vals[3]) if len(vals) > 3 else np.nan,
                            valid = True,
                        )
                        n_valid += 1
                    rows.append(row)
                    period = None

        print(f"valid={n_valid} invalid/9999={n_invalid}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index('start_time').sort_index()
    df.index.name = 'time'
    return df


# ---------------------------------------------------------------------------
# spectrum.log parser
# ---------------------------------------------------------------------------

def load_spectra(
    data_path:    Path,
    file_pattern: str = '*spectrum.log',
) -> xr.Dataset:
    """
    Parse all spectrum.log files under data_path.

    Returns an xarray Dataset with dimensions (time, frequency):
        energy_spectrum  [m²/Hz]   (time × frequency)
        direction        [deg]
        spread           [deg]
        a1, b1, a2, b2   [dimensionless]
        check            [dimensionless] — ratio vertical/horizontal motion
        valid            [bool]     — False = all-9999 burst
        Hs_from_spectrum [m]        — sqrt(variance) × 4, derived
        n_samples        [int]
        sampling_freq    [Hz]
        lat, lon         start position

    Bursts that are entirely 9999 are retained with NaN values and
    valid=False so gaps are visible in time series plots.
    """
    data_path = Path(data_path)
    files = sorted(data_path.rglob(file_pattern))
    if not files:
        warnings.warn(f"load_spectra: no files matching '{file_pattern}' under {data_path}")
        return xr.Dataset()

    bursts   = []   # list of dicts, one per PERIOD block
    freq_ref = None

    for fpath in files:
        print(f"  parsing {fpath.name} …", end=' ')
        n_valid = n_invalid = 0

        period = None
        current: dict = {}

        with open(fpath, 'r', errors='replace') as f:
            for line in f:
                line  = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                tag   = parts[0]

                if tag == 'PERIOD':
                    # Save previous burst if complete
                    if period is not None and 'energy' in current:
                        bursts.append({**period, **current})
                    period  = _parse_period(parts)
                    current = {}

                elif tag == 'ENERGYSPECTRA' and period is not None:
                    vals = _parse_row(tag, parts, 49)
                    current['valid']  = vals is not None
                    current['energy'] = vals if vals is not None else np.full(49, np.nan, dtype=np.float32)
                    if vals is None:
                        n_invalid += 1
                    else:
                        n_valid += 1

                elif tag == 'FREQUENCY' and period is not None:
                    vals = _parse_row(tag, parts, 49)
                    if vals is not None and freq_ref is None:
                        freq_ref = vals   # store once — fixed grid

                elif tag in ('DIR','SPREAD','A1','B1','A2','B2','CHECK') and period is not None:
                    key  = tag.lower()
                    vals = _parse_row(tag, parts, 49)
                    current[key] = vals if vals is not None else np.full(49, np.nan, dtype=np.float32)

        # Flush final burst
        if period is not None and 'energy' in current:
            bursts.append({**period, **current})

        print(f"valid={n_valid} invalid/9999={n_invalid}")

    if not bursts:
        return xr.Dataset()

    # Fixed frequency grid from manual / first file
    if freq_ref is None:
        # Reconstruct: 49 bins from 0.01172 Hz, step 0.00781 Hz
        freq_ref = 0.01171875 + np.arange(49) * 0.00781250

    n_freq = len(freq_ref)
    times  = pd.DatetimeIndex([b['start_time'] for b in bursts])

    def _stack(key, fill=np.nan):
        arr = np.stack([b.get(key, np.full(n_freq, fill, dtype=np.float32))
                        for b in bursts], axis=0)
        return arr.astype(np.float32)

    energy = _stack('energy')
    # Derived Hs from spectrum: 4 * sqrt(variance)
    # variance = integral(E * df) ≈ sum(E) * df
    df_val = np.median(np.diff(freq_ref))
    variance = np.nansum(energy, axis=1) * df_val
    Hs_spec  = 4.0 * np.sqrt(np.maximum(variance, 0))

    ds = xr.Dataset(
        {
            'energy_spectrum': (
                ['time', 'frequency'], energy,
                {'long_name': 'Wave energy spectrum',
                 'units': 'm2 Hz-1',
                 'standard_name': 'sea_surface_wave_variance_spectral_density'}
            ),
            'direction': (
                ['time', 'frequency'], _stack('dir'),
                {'long_name': 'Mean wave direction per frequency bin',
                 'units': 'degrees_true',
                 'comment': 'degrees from true north, nautical convention (coming from)'}
            ),
            'spread': (
                ['time', 'frequency'], _stack('spread'),
                {'long_name': 'Directional spread per frequency bin',
                 'units': 'degrees'}
            ),
            'a1': (['time', 'frequency'], _stack('a1'),
                   {'long_name': 'First directional moment, east-west', 'units': '1'}),
            'b1': (['time', 'frequency'], _stack('b1'),
                   {'long_name': 'First directional moment, north-south', 'units': '1'}),
            'a2': (['time', 'frequency'], _stack('a2'),
                   {'long_name': 'Second directional moment, east-west', 'units': '1'}),
            'b2': (['time', 'frequency'], _stack('b2'),
                   {'long_name': 'Second directional moment, north-south', 'units': '1'}),
            'check': (
                ['time', 'frequency'], _stack('check'),
                {'long_name': 'Spectral quality check factor',
                 'units': '1',
                 'comment': 'Ratio vertical/horizontal motion. Should be ~1 for deep water. '
                            'Large deviations indicate poor data quality.'}
            ),
            'valid': (
                ['time'], np.array([b.get('valid', False) for b in bursts]),
                {'long_name': 'Burst validity flag',
                 'comment': 'False = all-9999 burst (GPSWaves had no valid data)'}
            ),
            'Hs_from_spectrum': (
                ['time'], Hs_spec.astype(np.float32),
                {'long_name': 'Significant wave height from spectral integral',
                 'units': 'm',
                 'comment': '4 * sqrt(integral of energy spectrum)'}
            ),
            'n_samples': (
                ['time'], np.array([b.get('n_samples', 0) for b in bursts], dtype=np.int32),
                {'long_name': 'Number of samples used in burst'}
            ),
            'sampling_freq': (
                ['time'], np.array([b.get('sampling_freq', np.nan) for b in bursts], dtype=np.float32),
                {'long_name': 'Sampling frequency of GPS used for wave calculation',
                 'units': 'Hz'}
            ),
            'start_lat': (
                ['time'], np.array([b.get('start_lat', np.nan) for b in bursts], dtype=np.float64),
                {'long_name': 'Latitude at start of burst', 'units': 'degrees_north'}
            ),
            'start_lon': (
                ['time'], np.array([b.get('start_lon', np.nan) for b in bursts], dtype=np.float64),
                {'long_name': 'Longitude at start of burst', 'units': 'degrees_east'}
            ),
        },
        coords={
            'time':      times,
            'frequency': (
                'frequency', freq_ref.astype(np.float32),
                {'long_name': 'Wave frequency', 'units': 'Hz'}
            ),
        },
        attrs={
            'title':       'GPSWaves directional wave spectrum',
            'source':      'spectrum.log',
            'conventions': 'CF-1.8',
            'reference':   'Herbers et al., J. Atmos. Ocean. Tech., 2012',
            'check_note':  'CHECK ≈ 1 expected in deep water; large values flag bad data',
        }
    )
    return ds