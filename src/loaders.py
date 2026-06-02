"""
src/loaders.py
--------------
Thin data loaders for each Wave Glider sensor stream.
Wraps existing parsing logic (TOA5/TOB3 for CR6, ENS binary for ADCP,
and Marcel's xdata decoder for GPSWaves) into a uniform interface that
returns a pandas DataFrame with a proper DatetimeIndex.

Each loader returns None and prints a warning if no matching files are found,
so the notebook can continue gracefully with other sensors.
"""

from __future__ import annotations

import struct
import os
import datetime as _dt
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Utilities
# =============================================================================

def find_files(data_path: Path, pattern: str) -> list[Path]:
    """Return sorted list of files matching a glob pattern under data_path.
    Always searches recursively so nested folders (e.g. CR6/Raw Data/) are included.
    """
    return sorted(data_path.rglob(pattern))


def get_file_inventory(cfg: dict, data_path: Path) -> pd.DataFrame:
    """
    Scan for expected files and return a summary DataFrame.
    Used in Section 1 of the notebook.
    """
    rows = []

    sensor_map = {
        'rbr':  ('CR6',  cfg['cr6']['rbr']),
        'wxt':  ('CR6',  cfg['cr6']['wxt']),
        'gill': ('CR6',  cfg['cr6']['gill']),
        'spn1': ('CR6',  cfg['cr6']['spn1']),
        'cgr4': ('CR6',  cfg['cr6']['cgr4']),
        'adcp': ('ADCP', cfg['adcp']),
    }

    for key, (source, scfg) in sensor_map.items():
        if not scfg.get('enabled', True):
            rows.append({
                'sensor': scfg['label'], 'source': source,
                'found': '—', 'n_files': 0,
                'time_start': '—', 'time_end': '—', 'notes': 'disabled'
            })
            continue

        files = find_files(data_path, scfg['file_pattern'])
        fmt = _detect_format(files[0]) if files else '—'
        rows.append({
            'sensor':   scfg['label'],
            'source':   source,
            'found':    '✅' if files else '❌',
            'format':   fmt,
            'n_files':  len(files),
            'time_start': '—',
            'time_end':   '—',
            'notes':    '' if files else 'no files found'
        })

    if cfg['sensors'].get('gpswaves', False):
        rows.append({
            'sensor': 'GPSWaves / IMU', 'source': 'XDATA',
            'found': '?', 'format': '—', 'n_files': 0,
            'time_start': '—', 'time_end': '—',
            'notes': 'checked in section 4'
        })

    return pd.DataFrame(rows).set_index('sensor')


# =============================================================================
# Format detection
# =============================================================================

def _detect_format(filepath: Path) -> str:
    """
    Sniff the first bytes of a Campbell .dat file to determine format.
    Returns 'TOA5' or 'TOB3'.
    """
    with open(filepath, 'rb') as f:
        first = f.read(10).decode('latin-1', errors='replace')
    if 'TOA5' in first:
        return 'TOA5'
    if 'TOB3' in first:
        return 'TOB3'
    # Binary content without a recognisable ASCII tag → assume TOB3
    if any(ord(c) > 127 or ord(c) < 9 for c in first):
        return 'TOB3'
    raise ValueError(f"Unrecognised Campbell file format in {filepath.name}")


# =============================================================================
# TOB3 binary parser
# (Adapted from spirrobe/campbell, Unlicense. Hand-validated for this deployment.)
# =============================================================================

_TOB3_EPOCH = _dt.datetime(1990, 1, 1)


def _fp2_to_float(fp2: int) -> float:
    """Decode a Campbell FP2 (2-byte big-endian) integer to float."""
    if fp2 == 0x1fff: return float('inf')
    if fp2 == 0x9fff: return float('-inf')
    if fp2 == 0x9ffe: return float('nan')
    mantissa = fp2 & 0x1fff
    exponent = (fp2 & 0x6000) >> 13
    value = mantissa * (10 ** -exponent)
    return -value if (fp2 & 0x8000) else value


def _decode_fp2(b: bytes) -> float:
    (val,) = struct.unpack('>H', b)
    return _fp2_to_float(val)

def _decode_uint2(b: bytes) -> int:
    return struct.unpack('>H', b)[0]

def _make_ascii_decoder(n: int):
    def decode(b: bytes) -> str:
        return b.rstrip(b'\x00').decode('latin-1', errors='replace').strip()
    return decode

def _parse_interval(s: str) -> float:
    """Parse '30 SEC' / '1000 MSEC' / '50 MSEC' -> seconds."""
    n_str, unit = s.strip().split()
    n, unit = int(n_str), unit.upper()
    return {'USEC': n/1e6, 'MSEC': n/1e3, 'SEC': float(n), 'MIN': n*60.0}[unit]

def _parse_subsec_tick(s: str) -> float:
    """Parse 'Sec100Usec' -> 1e-4, etc."""
    s = s.strip()
    if not s.lower().startswith('sec'):
        return 1.0
    tail = s[3:]
    digits = ''.join(c for c in tail if c.isdigit())
    unit   = ''.join(c for c in tail if c.isalpha()).lower()
    n = int(digits) if digits else 1
    if 'usec' in unit: return n / 1e6
    if 'msec' in unit: return n / 1e3
    return float(n)


def _read_tob3(filepath: Path) -> pd.DataFrame:
    """
    Parse a single TOB3 binary file and return a DataFrame with a DatetimeIndex.
    Supports field types: FP2, UINT2, ASCII(n).
    """
    with open(filepath, 'rb') as f:
        header = []
        for _ in range(6):
            line = f.readline().rstrip().decode('latin-1').split(',')
            header.append([x.replace('"', '') for x in line])

        col_names  = header[2]
        field_types = [t.strip() for t in header[5]]

        field_decoders = []
        for ftype in field_types:
            if ftype == 'FP2':
                field_decoders.append((2, _decode_fp2))
            elif ftype == 'UINT2':
                field_decoders.append((2, _decode_uint2))
            elif ftype.startswith('ASCII('):
                n = int(ftype[6:-1])
                field_decoders.append((n, _make_ascii_decoder(n)))
            else:
                raise ValueError(f"Unsupported TOB3 field type: {ftype!r}")

        rec_size       = sum(sz for sz, _ in field_decoders)
        framesize      = int(header[1][2])
        val_stamp      = int(header[1][4])
        validation_set = {val_stamp, (2**16 - 1) - val_stamp}
        sub_step       = _parse_interval(header[1][1])
        sub_tick       = _parse_subsec_tick(header[1][5])

        fhdr_fmt  = '<LLL'
        ffoot_fmt = '<HH'
        fhdr_size  = struct.calcsize(fhdr_fmt)
        ffoot_size = struct.calcsize(ffoot_fmt)
        data_bytes = framesize - fhdr_size - ffoot_size
        n_rec      = data_bytes // rec_size

        file_size = os.path.getsize(filepath)
        timestamps, rows = [], []

        while True:
            if file_size - f.tell() < framesize:
                break
            frame_sec, frame_sub, frame_recnum = struct.unpack(fhdr_fmt, f.read(fhdr_size))
            raw = f.read(data_bytes)
            offset_flags, validation = struct.unpack(ffoot_fmt, f.read(ffoot_size))

            if validation not in validation_set:
                continue

            if offset_flags == 0:
                n_valid = n_rec
            else:
                bits = bin(offset_flags)[2:].zfill(16)
                valid_bytes = int(bits[4:], 2)
                n_valid = min(max(0, (valid_bytes - fhdr_size) // rec_size), n_rec)

            for k in range(n_valid):
                row, offset = [], k * rec_size
                for sz, decoder in field_decoders:
                    row.append(decoder(raw[offset:offset + sz]))
                    offset += sz
                rows.append(row)
                timestamps.append(
                    _TOB3_EPOCH + _dt.timedelta(
                        seconds=frame_sec + k * sub_step + frame_sub * sub_tick
                    )
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=col_names)
    df.index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name='TIMESTAMP')
    return df


# =============================================================================
# Unified CR6 loader  (autodetects TOA5 or TOB3 per file)
# =============================================================================

def load_cr6(
    data_path:   Path,
    file_pattern: str,
    test_period:  Optional[tuple] = None,
    label:        str = '',
) -> Optional[pd.DataFrame]:
    """
    Load CR6 sensor files matching file_pattern under data_path.
    Autodetects TOA5 (ASCII) or TOB3 (binary) format per file.
    Concatenates multiple files, drops duplicates, returns a DatetimeIndex DataFrame.

    If test_period is provided, records outside the window are reported and removed.
    """
    files = find_files(data_path, file_pattern)
    if not files:
        warnings.warn(f"load_cr6: no files found for '{file_pattern}' under {data_path}")
        return None

    frames = []
    for fpath in files:
        try:
            fmt = _detect_format(fpath)
            if fmt == 'TOA5':
                df = _read_toa5(fpath)
            else:
                df = _read_tob3(fpath)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            warnings.warn(f"load_cr6: failed to load {fpath.name}: {e}")

    if not frames:
        return None

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

    return apply_test_period(combined, test_period, label=label or file_pattern)


def _read_toa5(fpath: Path) -> pd.DataFrame:
    """Read a single TOA5 file into a DatetimeIndex DataFrame."""
    df = pd.read_csv(
        fpath,
        skiprows=[0, 2, 3],
        header=0,
        na_values=['NAN', 'NaN', '"NAN"'],
        low_memory=False,
    )
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    return df.set_index('TIMESTAMP').sort_index()


# Keep load_toa5 as a thin alias for backwards compatibility
def load_toa5(data_path: Path, file_pattern: str) -> Optional[pd.DataFrame]:
    """Backwards-compatible alias for load_cr6. Loads TOA5 files only."""
    return load_cr6(data_path, file_pattern)


# =============================================================================
# ADCP loader (stub)
# =============================================================================

def load_adcp(data_path: Path, file_pattern: str):
    """
    Load ADCP data. Stub — wire up decode_ens from waveglider-adcp-processing.
    """
    files = find_files(data_path, file_pattern)
    if not files:
        warnings.warn(f"load_adcp: no .ENS files found under {data_path}")
        return None

    raise NotImplementedError(
        "ADCP loader not yet wired up. "
        "Copy or import decode_ens from waveglider-adcp-processing."
    )


# =============================================================================
# =============================================================================

def load_gpswaves(data_path: Path) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load GPSWaves + IMU. Stub — wire up from waveglider-xdata.
    Returns (df_waves, df_imu).
    """
    raise NotImplementedError(
        "GPSWaves loader not yet wired up. "
        "Import from polar-gliders-operations/waveglider-xdata."
    )


# =============================================================================
# GPSWaves / IMU loader
# =============================================================================

def load_gpswaves(data_path: Path) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Find all raw.log files under data_path and parse MicroStrain MIP packets.

    Returns
    -------
    df_imu : ~20 Hz IMU data (accel, gyro, mag, euler angles)
    df_gps : ~1 Hz GPS data  (LLH position, NED velocity, fix info)
    """
    from src.gpswaves import load_gpswaves as _load
    return _load(data_path)


# =============================================================================
# Test period filter  (applied after loading, before returning to notebook)
# =============================================================================

def apply_test_period(
    df:         pd.DataFrame,
    test_period: Optional[tuple],
    label:      str = '',
) -> pd.DataFrame:
    """
    Filter a DataFrame to the configured test period.

    Records outside the window are reported but not silently dropped —
    a summary line is printed so the user can see what was removed and
    decide whether it looks like a clock/timezone issue or a corrupt record.

    Parameters
    ----------
    df          : DataFrame with DatetimeIndex (UTC-aware)
    test_period : (start, end) as pd.Timestamps, or None (no filtering)
    label       : sensor name for the printed report

    Returns
    -------
    Filtered DataFrame (may be identical to input if no outliers)
    """
    if df is None or df.empty or test_period is None:
        return df

    start, end = test_period

    # Ensure index is tz-aware UTC for comparison
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
        df  = df.copy()
        df.index = idx

    mask_inside  = (idx >= start) & (idx <= end)
    mask_outside = ~mask_inside

    n_outside = mask_outside.sum()
    if n_outside > 0:
        outside = df[mask_outside]
        earliest = outside.index.min()
        latest   = outside.index.max()
        print(
            f"  ⚠️  {label}: {n_outside} record(s) outside test period "
            f"[{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}]\n"
            f"     earliest outlier: {earliest}\n"
            f"     latest outlier:   {latest}\n"
            f"     → excluded from analysis"
        )

    return df[mask_inside]