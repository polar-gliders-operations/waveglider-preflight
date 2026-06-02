"""
src/gpswaves.py
---------------
Parser for GPSWaves raw.log files as recorded by the Wave Glider XDATA system.

The raw.log contains two record types:
  PACKET  — binary MicroStrain MIP payload (hex-encoded), one per IMU sample
  SYSGPS  — ASCII GPS position fix from the Wave Glider nav system

MIP packets come in two descriptor sets:
  0x80 — IMU data  (accel, gyro, mag, euler angles, GPS-correlated timestamp)
  0x81 — GPS data  (LLH position, NED velocity, fix info, GPS timestamp)

Field descriptors decoded:
  IMU 0x04  Scaled Accelerometer  3x float32 BE  [g]
  IMU 0x05  Scaled Gyro           3x float32 BE  [rad/s]
  IMU 0x06  Scaled Magnetometer   3x float32 BE  [Gauss]
  IMU 0x0A  Euler Angles          3x float32 BE  [rad] + uint16 + uint16 flags
  IMU 0x11  Stabilised Accel      3x float32 BE  [g]   (low-pass filtered)
  IMU 0x12  GPS Corr. Timestamp   double BE tow [s] + uint16 week + uint16 flags
  GPS 0x03  LLH Position          4x double BE (lat,lon,alt_ellipsoid,alt_msl) [deg/m]
                                  + 2x float32 (hacc,vacc) [m] + uint16 flags
  GPS 0x05  NED Velocity          8x float32 BE (vn,ve,vd,speed,gspeed,heading,
                                  speed_acc,heading_acc) + uint16 flags
  GPS 0x08  GPS Fix Info          uint8 fix_type + uint8 num_sv + ...
  GPS 0x09  GPS Timestamp         double BE tow [s] + uint16 week + uint16 flags

Public API
----------
  parse_raw_log(path)     -> (df_imu, df_gps, df_sysgps, df_errors)
  load_gpswaves(data_path) -> (df_imu, df_gps)  [for use by loaders.py]

All returned DataFrames have a UTC DatetimeIndex named 'time'.
"""

from __future__ import annotations

import hashlib
import math
import struct
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# GPS epoch
# ---------------------------------------------------------------------------
_GPS_EPOCH = pd.Timestamp("1980-01-06", tz="UTC")

_G = 9.80665  # m/s² per g


def _gps_to_utc(tow_s: float, week: int) -> pd.Timestamp:
    """Convert GPS week + time-of-week to UTC (approximate, ignores leap seconds)."""
    return _GPS_EPOCH + pd.Timedelta(weeks=int(week)) + pd.Timedelta(seconds=float(tow_s))


# ---------------------------------------------------------------------------
# MIP field decoders
# ---------------------------------------------------------------------------

def _decode_imu_packet(data: bytes, ts_log: str) -> dict:
    """
    Decode a MIP descriptor-set 0x80 (IMU) packet payload into a flat dict.
    ts_log is the ISO timestamp from the log line (used as fallback).
    """
    rec = {}  # time_log removed — ts_log used only as fallback for index
    offset = 0
    while offset < len(data):
        if offset + 2 > len(data):
            break
        field_len  = data[offset]
        field_desc = data[offset + 1]
        if field_len < 2:
            break
        field_data = data[offset + 2 : offset + field_len]
        offset += field_len

        try:
            if field_desc == 0x04 and len(field_data) >= 12:
                ax, ay, az = struct.unpack(">fff", field_data[:12])
                rec["accel_x_g"]  = ax
                rec["accel_y_g"]  = ay
                rec["accel_z_g"]  = az
                rec["accel_x"]    = ax * _G
                rec["accel_y"]    = ay * _G
                rec["accel_z"]    = az * _G

            elif field_desc == 0x05 and len(field_data) >= 12:
                gx, gy, gz = struct.unpack(">fff", field_data[:12])
                rec["gyro_x"]     = gx          # rad/s
                rec["gyro_y"]     = gy
                rec["gyro_z"]     = gz
                rec["gyro_x_deg"] = math.degrees(gx)
                rec["gyro_y_deg"] = math.degrees(gy)
                rec["gyro_z_deg"] = math.degrees(gz)

            elif field_desc == 0x06 and len(field_data) >= 12:
                mx, my, mz = struct.unpack(">fff", field_data[:12])
                rec["mag_x"] = mx
                rec["mag_y"] = my
                rec["mag_z"] = mz

            elif field_desc == 0x0A and len(field_data) >= 12:
                roll, pitch, yaw = struct.unpack(">fff", field_data[:12])
                rec["roll_rad"]  = roll
                rec["pitch_rad"] = pitch
                rec["yaw_rad"]   = yaw
                rec["roll_deg"]  = math.degrees(roll)
                rec["pitch_deg"] = math.degrees(pitch)
                rec["yaw_deg"]   = math.degrees(yaw)

            elif field_desc == 0x11 and len(field_data) >= 12:
                sx, sy, sz = struct.unpack(">fff", field_data[:12])
                rec["stab_accel_x"] = sx
                rec["stab_accel_y"] = sy
                rec["stab_accel_z"] = sz

            elif field_desc == 0x12 and len(field_data) >= 12:
                tow, week, flags = struct.unpack(">dHH", field_data[:12])
                rec["gps_tow"]   = tow
                rec["gps_week"]  = week
                rec["gps_flags"] = flags
                if week > 0:
                    rec["time_gps"] = _gps_to_utc(tow, week)

        except struct.error:
            pass  # truncated field — skip

    return rec


def _decode_gps_packet(data: bytes, ts_log: str) -> dict:
    """
    Decode a MIP descriptor-set 0x81 (GPS) packet payload into a flat dict.
    """
    rec = {}  # time_log removed — ts_log used only as fallback for index
    offset = 0
    while offset < len(data):
        if offset + 2 > len(data):
            break
        field_len  = data[offset]
        field_desc = data[offset + 1]
        if field_len < 2:
            break
        field_data = data[offset + 2 : offset + field_len]
        offset += field_len

        try:
            if field_desc == 0x03 and len(field_data) >= 42:
                lat, lon, alt_e, alt_msl = struct.unpack(">dddd", field_data[:32])
                hacc, vacc = struct.unpack(">ff", field_data[32:40])
                flags = struct.unpack(">H", field_data[40:42])[0]
                rec["latitude"]     = lat
                rec["longitude"]    = lon
                rec["alt_ellipsoid"]= alt_e
                rec["alt_msl"]      = alt_msl
                rec["hacc"]         = hacc
                rec["vacc"]         = vacc
                rec["llh_flags"]    = flags

            elif field_desc == 0x05 and len(field_data) >= 34:
                vn, ve, vd, speed, gspeed, heading, sp_acc, hd_acc = struct.unpack(
                    ">ffffffff", field_data[:32]
                )
                flags = struct.unpack(">H", field_data[32:34])[0]
                rec["vel_north"]    = vn
                rec["vel_east"]     = ve
                rec["vel_down"]     = vd
                rec["speed"]        = speed
                rec["groundspeed"]  = gspeed
                rec["heading"]      = math.degrees(heading) if abs(heading) < 100 else heading
                rec["speed_acc"]    = sp_acc
                rec["heading_acc"]  = hd_acc
                rec["ned_flags"]    = flags

            elif field_desc == 0x08 and len(field_data) >= 6:
                # Field layout (GX4/GX5 firmware):
                # uint16 unknown/flags, uint8 fix_type, uint8 num_sv, uint16 fix_flags, ...
                # fix_type encoding: 0=No Fix, 1=2D, 2=3D, 3=SBAS,
                #                    4=Differential, 5=Dead Reckoning,
                #                    6=RTK Float, 7=RTK Fixed
                _, fix_type, num_sv, fix_flags = struct.unpack(">HBBH", field_data[:6])
                rec["fix_type"] = fix_type
                rec["num_sv"]   = num_sv
                rec["fix_flags"]= fix_flags

            elif field_desc == 0x09 and len(field_data) >= 12:
                tow, week, flags = struct.unpack(">dHH", field_data[:12])
                rec["gps_tow"]  = tow
                rec["gps_week"] = week
                if week > 0:
                    rec["time_gps"] = _gps_to_utc(tow, week)

        except struct.error:
            pass

    return rec


# ---------------------------------------------------------------------------
# Log line timestamp parser
# ---------------------------------------------------------------------------

def _parse_log_ts(date_str: str, time_str: str, micros_str: str) -> pd.Timestamp:
    ts_str = f"{date_str} {time_str}.{int(micros_str):06d}"
    return pd.Timestamp(ts_str, tz="UTC")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_raw_log(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse a single GPSWaves raw.log file.

    Returns
    -------
    df_imu    : IMU records (set 0x80) — one row per MIP packet, ~20 Hz
    df_gps    : GPS records (set 0x81) — one row per GPS packet, ~1 Hz
    df_sysgps : SYSGPS records (Wave Glider nav GPS), ~1 Hz
    df_errors : Lines that failed to parse
    """
    imu_rows    = []
    gps_rows    = []
    sysgps_rows = []
    error_rows  = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")

            try:
                rec_type = parts[0]

                if rec_type == "PACKET":
                    if len(parts) < 5:
                        raise ValueError(f"Too few fields: {len(parts)}")

                    ts_log   = _parse_log_ts(parts[1], parts[2], parts[3])
                    hex_str  = parts[4].strip()
                    payload  = bytes.fromhex(hex_str)

                    # MIP header: sync 0x75 0x65, descriptor set, payload_len
                    if len(payload) < 4 or payload[0] != 0x75 or payload[1] != 0x65:
                        raise ValueError("Not a valid MIP packet (bad sync bytes)")

                    desc_set    = payload[2]
                    payload_len = payload[3]
                    mip_payload = payload[4 : 4 + payload_len]

                    if desc_set == 0x80:
                        rec = _decode_imu_packet(mip_payload, ts_log)
                        rec["time"] = rec.pop("time_gps", ts_log)
                        imu_rows.append(rec)

                    elif desc_set == 0x81:
                        rec = _decode_gps_packet(mip_payload, ts_log)
                        rec["time"] = rec.pop("time_gps", ts_log)
                        gps_rows.append(rec)

                    else:
                        error_rows.append({
                            "line": line_num,
                            "type": "PACKET",
                            "error": f"Unknown descriptor set 0x{desc_set:02X}",
                            "raw": raw_line.rstrip(),
                        })

                elif rec_type == "SYSGPS":
                    if len(parts) < 6:
                        raise ValueError(f"Too few fields: {len(parts)}")
                    ts_log = _parse_log_ts(parts[1], parts[2], parts[3])
                    sysgps_rows.append({
                        "time":      ts_log,
                        "latitude":  float(parts[4]),
                        "longitude": float(parts[5]),
                    })

                else:
                    error_rows.append({
                        "line": line_num,
                        "type": rec_type,
                        "error": "Unknown record type",
                        "raw": raw_line.rstrip(),
                    })

            except Exception as e:
                error_rows.append({
                    "line":  line_num,
                    "type":  parts[0] if parts else "",
                    "error": str(e),
                    "raw":   raw_line.rstrip(),
                })

    def _to_df(rows: list, time_col: str = "time") -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if time_col in df.columns:
            df = df.set_index(time_col).sort_index()
        return df

    df_imu    = _to_df(imu_rows)
    df_gps    = _to_df(gps_rows)
    df_sysgps = _to_df(sysgps_rows)
    df_errors = pd.DataFrame(error_rows) if error_rows else pd.DataFrame()

    return df_imu, df_gps, df_sysgps, df_errors


# ---------------------------------------------------------------------------
# Multi-file loader (entry point for loaders.py)
# ---------------------------------------------------------------------------

def load_gpswaves(
    data_path: Path,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Find all raw.log files under data_path (recursive) and parse them.

    Returns
    -------
    df_imu : Combined IMU DataFrame (~20 Hz), DatetimeIndex
    df_gps : Combined GPS DataFrame (~1 Hz), DatetimeIndex

    SYSGPS and error records are printed as a summary but not returned —
    add them to the return signature if needed later.
    """
    log_files = sorted(data_path.rglob("*raw.log"))
    if not log_files:
        warnings.warn(f"load_gpswaves: no raw.log files found under {data_path}")
        return None, None

    print(f"load_gpswaves: found {len(log_files)} raw.log file(s)")

    all_imu    = []
    all_gps    = []
    total_errs = 0

    for fpath in log_files:
        print(f"  parsing {fpath.name} …", end=" ", flush=True)
        try:
            df_imu, df_gps, df_sysgps, df_errors = parse_raw_log(fpath)
            print(f"IMU={len(df_imu)} GPS={len(df_gps)} SYSGPS={len(df_sysgps)} errors={len(df_errors)}")
            if not df_imu.empty:
                all_imu.append(df_imu)
            if not df_gps.empty:
                all_gps.append(df_gps)
            total_errs += len(df_errors)
        except Exception as e:
            print(f"FAILED: {e}")
            warnings.warn(f"load_gpswaves: failed to parse {fpath.name}: {e}")

    if total_errs > 0:
        warnings.warn(f"load_gpswaves: {total_errs} parse errors across all files")

    df_imu_out = pd.concat(all_imu).sort_index() if all_imu else None
    df_gps_out = pd.concat(all_gps).sort_index() if all_gps else None

    return df_imu_out, df_gps_out