"""
src/adcp.py
-----------
Parser for Teledyne RDI WorkHorse ADCP binary .ENS files.

Extracted and refactored from waveglider_adcp_processing-WAM.ipynb.
Reads the binary file at the byte level without pycurrents — the standard
pycurrents reader misaligns on Wave Glider .ENS files.

Public API
----------
    parse_ens(path, sampling_interval_s, min_gap_s, main_segment) -> dict of xr.Dataset
    load_adcp(data_path, file_pattern, ...) -> dict of xr.Dataset

The returned dict has keys 'beam' and/or 'ENU' depending on what coordinate
systems are present in the file.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Block ID lookup
# =============================================================================

_BLOCK_IDS = {
    0x0000: 'FixedLeader',
    0x0080: 'VariableLeader',
    0x0100: 'Velocity',
    0x0200: 'Correlation',
    0x0300: 'EchoIntensity',
    0x0400: 'PercentGood',
    0x0600: 'BottomTrack',
    0x2000: 'Navigation',
}


def _label_from_ex(ex_value: int) -> str:
    transform_bits = (ex_value >> 3) & 0b11
    return {0b00: 'beam', 0b01: 'instrument', 0b10: 'ship', 0b11: 'ENU'}.get(
        transform_bits, f'unknown_0x{ex_value:02x}'
    )


# =============================================================================
# Step 1 — Read binary
# =============================================================================

def _read_raw(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


# =============================================================================
# Step 2 — Find ensemble boundaries
# =============================================================================

def _find_ensemble_markers(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Locate 0x7f7f markers and determine ensemble size from the most common spacing.
    Returns (real_markers, ensemble_size).
    """
    candidates = np.where((raw[:-1] == 0x7f) & (raw[1:] == 0x7f))[0]
    spacings   = np.diff(candidates)
    vals, counts = np.unique(spacings, return_counts=True)
    ensemble_size = int(vals[np.argmax(counts)])
    real_markers  = candidates[np.where(spacings == ensemble_size)[0]]
    return real_markers, ensemble_size


# =============================================================================
# Step 3 — Parse ensemble header → offset dict
# =============================================================================

def _parse_offset_dict(raw: np.ndarray, marker: int, ensemble_size: int) -> dict:
    """Read the first ensemble's header to build a block-name → byte-offset dict."""
    n_types = int(raw[marker + 5])
    offsets = []
    for i in range(n_types):
        o = int(raw[marker + 6 + i*2]) + int(raw[marker + 7 + i*2]) * 256
        offsets.append(o)

    offset_dict = {}
    for o in offsets:
        block_id = int(raw[marker + o]) + int(raw[marker + o + 1]) * 256
        name = _BLOCK_IDS.get(block_id, f'Unknown_0x{block_id:04x}')
        offset_dict[name] = o
    return offset_dict


# =============================================================================
# Step 4 — Read FixedLeader config
# =============================================================================

def _read_fixed_leader(raw: np.ndarray, marker: int, offset_dict: dict) -> dict:
    fl = offset_dict['FixedLeader']
    c  = raw[marker:]
    nbeams    = int(c[fl + 8])
    ncells    = int(c[fl + 9])
    npings    = int(c[fl + 10]) + int(c[fl + 11]) * 256
    cell_size = (int(c[fl + 12]) + int(c[fl + 13]) * 256) * 0.01   # cm → m
    blank     = (int(c[fl + 14]) + int(c[fl + 15]) * 256) * 0.01
    bin1_dist = (int(c[fl + 32]) + int(c[fl + 33]) * 256) * 0.01
    pulse_len = (int(c[fl + 34]) + int(c[fl + 35]) * 256) * 0.01
    bin_depths = bin1_dist + np.arange(ncells) * cell_size
    return dict(nbeams=nbeams, ncells=ncells, npings=npings,
                cell_size=cell_size, blank=blank,
                bin1_dist=bin1_dist, pulse_len=pulse_len,
                bin_depths=bin_depths)


# =============================================================================
# Step 5 — Parse timestamps
# =============================================================================

def _parse_timestamps(
    raw: np.ndarray,
    markers: np.ndarray,
    ensemble_size: int,
    offset_dict: dict,
) -> list[Optional[datetime]]:
    vl = offset_dict['VariableLeader']
    timestamps = []
    for idx in markers:
        try:
            c = raw[idx:idx + ensemble_size]
            ts = datetime(
                2000 + int(c[vl + 4]),
                int(c[vl + 5]),
                int(c[vl + 6]),
                int(c[vl + 7]),
                int(c[vl + 8]),
                int(c[vl + 9]),
                int(c[vl + 10]) * 10000,
            )
            timestamps.append(ts)
        except Exception:
            timestamps.append(None)
    return timestamps


# =============================================================================
# Step 6 — Segment selection
# =============================================================================

def _select_main_segment(
    timestamps: list,
    markers: np.ndarray,
    min_gap_s: float,
    main_segment: int,
) -> tuple[np.ndarray, list]:
    """
    Split on gaps > min_gap_s and return the requested segment's markers + timestamps.
    main_segment: -1 = last, 0 = first, 1 = second, etc.
    """
    valid = [(i, t) for i, t in enumerate(timestamps) if t is not None]
    if not valid:
        raise ValueError("No valid timestamps found in file")

    t_idx = [v[0] for v in valid]
    t_arr = pd.DatetimeIndex([v[1] for v in valid])
    dt    = np.diff(t_arr.asi8) / 1e9

    breaks = np.where(dt > min_gap_s)[0]
    if len(breaks) == 0:
        return markers, timestamps

    # Build segment index list
    seg_starts = [0] + list(breaks + 1)
    seg_ends   = list(breaks + 1) + [len(t_idx)]

    chosen = main_segment % len(seg_starts)
    s, e   = seg_starts[chosen], seg_ends[chosen]

    orig_s = t_idx[s]
    orig_e = t_idx[e - 1] + 1

    return markers[orig_s:orig_e], timestamps[orig_s:orig_e]


# =============================================================================
# Step 7 — Extract all data arrays for one segment
# =============================================================================

def _extract_segment(
    raw:           np.ndarray,
    markers:       np.ndarray,
    timestamps:    list,
    offset_dict:   dict,
    ensemble_size: int,
    fl_cfg:        dict,
) -> dict:
    n      = len(markers)
    ncells = fl_cfg['ncells']
    nbeams = fl_cfg['nbeams']

    t_seg  = pd.DatetimeIndex([t for t in timestamps if t is not None])

    vel = np.full((n, ncells, nbeams), np.nan, dtype=np.float32)
    cor = np.full((n, ncells, nbeams), np.nan, dtype=np.float32)
    ech = np.full((n, ncells, nbeams), np.nan, dtype=np.float32)
    pg  = np.full((n, ncells, 4),      np.nan, dtype=np.float32)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    sos      = np.full(n, np.nan)
    heading  = np.full(n, np.nan)
    pitch    = np.full(n, np.nan)
    roll     = np.full(n, np.nan)
    temp     = np.full(n, np.nan)
    pressure = np.full(n, np.nan)

    vel_off = offset_dict['Velocity']
    cor_off = offset_dict.get('Correlation')
    ech_off = offset_dict.get('EchoIntensity')
    pg_off  = offset_dict.get('PercentGood')
    nav_off = offset_dict.get('Navigation')
    vl_off  = offset_dict['VariableLeader']
    nb      = ncells * nbeams

    for i, idx in enumerate(markers):
        c = raw[idx:idx + ensemble_size]

        # Velocity
        v = np.frombuffer(c[vel_off+2 : vel_off+2+nb*2], dtype='<i2')
        v = v.reshape(ncells, nbeams).astype(np.float32)
        v[v == -32768] = np.nan
        vel[i] = v

        # Correlation, echo, percent good
        if cor_off:
            cor[i] = np.frombuffer(c[cor_off+2:cor_off+2+nb],
                                   dtype=np.uint8).reshape(ncells, nbeams).astype(np.float32)
        if ech_off:
            ech[i] = np.frombuffer(c[ech_off+2:ech_off+2+nb],
                                   dtype=np.uint8).reshape(ncells, nbeams).astype(np.float32)
        if pg_off:
            pg[i] = np.frombuffer(c[pg_off+2:pg_off+2+ncells*4],
                                  dtype=np.uint8).reshape(ncells, 4).astype(np.float32)

        # Navigation (BAM4 GPS)
        if nav_off:
            lat[i] = float(np.frombuffer(c[nav_off+14:nav_off+18], dtype='<i4')[0]) * 180.0 / 2**31
            lon[i] = float(np.frombuffer(c[nav_off+18:nav_off+22], dtype='<i4')[0]) * 180.0 / 2**31

        # VariableLeader scalars
        sos[i]      = int(c[vl_off+14]) + int(c[vl_off+15]) * 256
        heading[i]  = (int(c[vl_off+18]) + int(c[vl_off+19]) * 256) * 0.01
        pitch[i]    = np.frombuffer(c[vl_off+20:vl_off+22], dtype='<i2')[0] * 0.01
        roll[i]     = np.frombuffer(c[vl_off+22:vl_off+24], dtype='<i2')[0] * 0.01
        temp[i]     = np.frombuffer(c[vl_off+26:vl_off+28], dtype='<i2')[0] * 0.01
        pressure[i] = np.frombuffer(c[vl_off+48:vl_off+52], dtype='<i4')[0] * 0.001

    # Clean GPS
    lat[(lat < -90)  | (lat > 90)]  = np.nan
    lon[(lon < -180) | (lon > 180)] = np.nan
    lat[lat == 0] = np.nan
    lon[lon == 0] = np.nan

    return dict(t=t_seg, vel=vel, cor=cor, ech=ech, pg=pg,
                lat=lat, lon=lon, sos=sos, heading=heading,
                pitch=pitch, roll=roll, temp=temp, pressure=pressure)


# =============================================================================
# Step 8 — Build xarray Dataset
# =============================================================================

def _build_dataset(
    data:              dict,
    coordinate_system: str,
    fl_cfg:            dict,
    source_file:       str,
    sampling_interval_s: int,
) -> xr.Dataset:
    bin_depths = fl_cfg['bin_depths']
    ncells     = fl_cfg['ncells']
    nbeams     = fl_cfg['nbeams']

    # ENU data: the 4 velocity columns are East, North, Up, Error — not beam radials.
    # Use a separate 'component' dimension so it's unambiguous. Correlation / echo /
    # percent_good are always stored in beam coordinates regardless of EX setting.
    is_enu = (coordinate_system == 'ENU')
    vel_dim      = 'component' if is_enu else 'beam'
    vel_longname = ('ADCP velocity (ENU — East/North/Up/Error)'
                    if is_enu else
                    f'ADCP velocity ({coordinate_system} coordinates)')
    vel_comment  = ('East, North, Up, Error velocity components'
                    if is_enu else
                    f'Coordinate system: {coordinate_system}')

    ds = xr.Dataset(
        {
            'velocity': (
                ['time', 'cell', vel_dim], data['vel'],
                {'long_name': vel_longname,
                 'units': 'mm s-1',
                 'comment': vel_comment}
            ),
            'correlation': (
                ['time', 'cell', 'beam'], data['cor'],
                {'long_name': 'Beam correlation magnitude', 'units': 'counts',
                 'comment': 'Values below 64 indicate poor data quality'}
            ),
            'echo_intensity': (
                ['time', 'cell', 'beam'], data['ech'],
                {'long_name': 'Echo intensity', 'units': 'counts',
                 'comment': '0.45 dB per count'}
            ),
            'percent_good': (
                ['time', 'cell', 'pg_type'], data['pg'],
                {'long_name': 'Percent good', 'units': 'percent',
                 'comment': 'pg_type 4 = all 4 beams good (use this one)'}
            ),
            'latitude':  (['time'], data['lat'],
                          {'long_name': 'latitude',  'units': 'degrees_north',
                           'standard_name': 'latitude'}),
            'longitude': (['time'], data['lon'],
                          {'long_name': 'longitude', 'units': 'degrees_east',
                           'standard_name': 'longitude'}),
            'heading':   (['time'], data['heading'],
                          {'long_name': 'instrument heading', 'units': 'degrees',
                           'comment': 'magnetic, not corrected for declination'}),
            'pitch':     (['time'], data['pitch'],
                          {'long_name': 'pitch', 'units': 'degrees',
                           'comment': 'positive = bow up'}),
            'roll':      (['time'], data['roll'],
                          {'long_name': 'roll', 'units': 'degrees',
                           'comment': 'positive = port up'}),
            'temperature': (['time'], data['temp'],
                            {'long_name': 'transducer temperature',
                             'units': 'degrees_C',
                             'standard_name': 'sea_water_temperature',
                             'comment': 'good proxy for SST on surface platform'}),
            'pressure':  (['time'], data['pressure'],
                          {'long_name': 'instrument pressure', 'units': 'dbar',
                           'standard_name': 'sea_water_pressure'}),
            'speed_of_sound': (['time'], data['sos'],
                               {'long_name': 'speed of sound', 'units': 'm s-1'}),
        },
        coords={
            'time':    data['t'],
            'cell':    np.arange(ncells),
            'range':   (['cell'], bin_depths,
                        {'long_name': 'range from transducer', 'units': 'm'}),
            # 'beam' is always present (used by correlation / echo / percent_good).
            # 'component' is added for ENU datasets to label the velocity dimension.
            'beam':      np.arange(1, nbeams + 1),
            **({'component': ['East', 'North', 'Up', 'Error']}
               if is_enu else {}),
            'pg_type': np.arange(1, 5),
        },
        attrs={
            'title':               f'WaveGlider ADCP — {coordinate_system} coordinates',
            'instrument':          'Teledyne RDI WorkHorse',
            'platform':            'Wave Glider',
            'source_file':         source_file,
            'coordinate_system':   coordinate_system,
            'deployment_start':    str(data['t'][0]),
            'deployment_end':      str(data['t'][-1]),
            'ncells':              ncells,
            'cell_size_m':         fl_cfg['cell_size'],
            'bin1_distance_m':     fl_cfg['bin1_dist'],
            'pulse_length_m':      fl_cfg['pulse_len'],
            'npings_per_ensemble': fl_cfg['npings'],
            'sampling_interval_s': sampling_interval_s,
            'magnetic_declination_note': 'not applied',
            'conventions':         'CF-1.8',
            'created':             pd.Timestamp.now().isoformat(),
        }
    )
    return ds


# =============================================================================
# Public API
# =============================================================================

def parse_ens(
    path:                Path,
    sampling_interval_s: int   = 1,
    min_gap_s:           float = 3600.0,
    main_segment:        int   = -1,
) -> dict[str, xr.Dataset]:
    """
    Parse a Teledyne RDI WorkHorse .ENS binary file into xarray Datasets.

    Parameters
    ----------
    path                : path to the .ENS file
    sampling_interval_s : expected seconds between ensembles (1 = 1 Hz)
    min_gap_s           : gaps larger than this separate deployment segments
    main_segment        : which segment to keep (-1 = last, 0 = first, ...)

    Returns
    -------
    dict with keys 'beam' and/or 'ENU', each an xr.Dataset
    """
    path = Path(path)
    print(f"Parsing {path.name}  ({path.stat().st_size / 1e6:.1f} MB) …")

    raw = _read_raw(path)

    markers, ensemble_size = _find_ensemble_markers(raw)
    print(f"  Ensembles found: {len(markers):,}  (ensemble size: {ensemble_size} bytes)")

    offset_dict = _parse_offset_dict(raw, markers[0], ensemble_size)
    fl_cfg      = _read_fixed_leader(raw, markers[0], offset_dict)
    print(f"  Config: {fl_cfg['ncells']} cells × {fl_cfg['nbeams']} beams, "
          f"cell size {fl_cfg['cell_size']:.2f} m, "
          f"bin1 {fl_cfg['bin1_dist']:.2f} m")

    timestamps = _parse_timestamps(raw, markers, ensemble_size, offset_dict)
    valid_ts   = [t for t in timestamps if t is not None]
    print(f"  Timestamps: {min(valid_ts)} → {max(valid_ts)}")

    main_markers, main_timestamps = _select_main_segment(
        timestamps, markers, min_gap_s, main_segment
    )
    print(f"  Main segment: {len(main_markers):,} ensembles")

    # Detect coordinate system(s)
    fl   = offset_dict['FixedLeader']
    ex_all = np.array([int(raw[idx + fl + 25]) for idx in main_markers])
    unique_ex = np.unique(ex_all)

    ex_changes = np.where(np.diff(ex_all) != 0)[0]
    if len(ex_changes) == 0:
        label = _label_from_ex(ex_all[0])
        print(f"  Coordinate system: {label}")
        segments = {label: (main_markers, main_timestamps)}
    else:
        split = ex_changes[0] + 1
        label_a = _label_from_ex(ex_all[0])
        label_b = _label_from_ex(ex_all[split])
        print(f"  Coordinate change at ensemble {split}: {label_a} → {label_b}")
        segments = {
            label_a: (main_markers[:split], main_timestamps[:split]),
            label_b: (main_markers[split:], main_timestamps[split:]),
        }

    # Extract each segment
    datasets = {}
    for label, (seg_markers, seg_timestamps) in segments.items():
        print(f"  Extracting {label} segment ({len(seg_markers):,} ensembles) …")
        data = _extract_segment(
            raw, seg_markers, seg_timestamps,
            offset_dict, ensemble_size, fl_cfg
        )
        datasets[label] = _build_dataset(
            data, label, fl_cfg,
            source_file=path.name,
            sampling_interval_s=sampling_interval_s,
        )
        t = data['t']
        print(f"    Temperature: {np.nanmin(data['temp']):.2f}–{np.nanmax(data['temp']):.2f} °C")
        print(f"    Pressure:    {np.nanmin(data['pressure']):.2f}–{np.nanmax(data['pressure']):.2f} dbar")
        print(f"    Heading:     {np.nanmin(data['heading']):.1f}–{np.nanmax(data['heading']):.1f} °")

    return datasets


def load_adcp(
    data_path:           Path,
    file_pattern:        str   = '*.ENS',
    test_period:         Optional[tuple] = None,
    sampling_interval_s: int   = 1,
    min_gap_s:           float = 3600.0,
    main_segment:        int   = -1,
) -> dict[str, xr.Dataset]:
    """
    Find all .ENS files under data_path and parse them.
    If test_period is given, the returned datasets are clipped to that window.

    Returns dict with keys 'beam' and/or 'ENU'.
    """
    from src.loaders import find_files, apply_test_period

    files = find_files(data_path, file_pattern)
    if not files:
        warnings.warn(f"load_adcp: no files matching '{file_pattern}' under {data_path}")
        return {}

    all_datasets: dict[str, list] = {}
    for fpath in files:
        try:
            result = parse_ens(fpath, sampling_interval_s, min_gap_s, main_segment)
            for label, ds in result.items():
                all_datasets.setdefault(label, []).append(ds)
        except Exception as e:
            warnings.warn(f"load_adcp: failed on {fpath.name}: {e}")

    # Concatenate multiple files per coordinate system
    combined = {}
    for label, ds_list in all_datasets.items():
        ds = xr.concat(ds_list, dim='time').sortby('time') if len(ds_list) > 1 else ds_list[0]
        if test_period:
            start, end = test_period
            # ADCP timestamps are tz-naive UTC — strip tz from test_period for comparison
            start_naive = pd.Timestamp(start).tz_convert('UTC').tz_localize(None)
            end_naive   = pd.Timestamp(end).tz_convert('UTC').tz_localize(None)
            n_before = len(ds.time)
            ds = ds.sel(time=slice(start_naive, end_naive))
            n_out = n_before - len(ds.time)
            if n_out > 0:
                print(f"  ⚠️  ADCP ({label}): {n_out} records outside test period — excluded")
        combined[label] = ds

    return combined