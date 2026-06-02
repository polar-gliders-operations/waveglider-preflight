"""
src/diagnostics.py
------------------
Diagnostic functions called by the preflight notebook.
Each check returns a dict that feeds into the final summary table.

Return format for all check_* functions:
    {
        'sensor':  str,     # human-readable sensor label
        'check':   str,     # what was checked
        'status':  str,     # '✅' | '⚠️' | '❌' | '—'
        'detail':  str,     # one-line description of the result
    }
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from IPython.display import display, HTML


STATUS_PASS = '✅'
STATUS_WARN = '⚠️'
STATUS_FAIL = '❌'
STATUS_SKIP = '—'


# =============================================================================
# Sampling rate check
# =============================================================================

def check_sampling_rate(df: Optional[pd.DataFrame], scfg: dict, label: str) -> dict:
    """
    Compute the median sampling interval and compare to the expected value.
    Also prints a brief summary.
    """
    check_name = 'Sampling rate'

    if df is None or df.empty:
        print(f"[{STATUS_FAIL}] {label} — {check_name}: no data loaded")
        return {'sensor': label, 'check': check_name, 'status': STATUS_FAIL,
                'detail': 'no data'}

    dt = df.index.to_series().diff().dt.total_seconds().dropna()
    median_dt = dt.median()
    expected  = scfg['expected_interval_s']
    tol       = scfg['tolerance_s']
    error     = abs(median_dt - expected)

    if error <= tol:
        status = STATUS_PASS
    elif error <= tol * 3:
        status = STATUS_WARN
    else:
        status = STATUS_FAIL

    gap_threshold = expected * 3
    gaps = dt[dt > gap_threshold]
    gap_info = f", {len(gaps)} gap(s) > {gap_threshold:.1f}s" if len(gaps) > 0 else ""

    detail = (f"median Δt = {median_dt:.4f}s (expected {expected}s ± {tol}s)"
              f"{gap_info}")

    print(f"[{status}] {label} — {check_name}: {detail}")
    return {'sensor': label, 'check': check_name, 'status': status, 'detail': detail}


# =============================================================================
# Variable range checks
# =============================================================================

def check_variable_ranges(df: Optional[pd.DataFrame], scfg: dict, label: str) -> dict:
    """
    Check that configured variables exist and fall within expected ranges.
    """
    check_name = 'Variable ranges'

    if df is None or df.empty:
        return {'sensor': label, 'check': check_name, 'status': STATUS_FAIL,
                'detail': 'no data'}

    issues = []
    missing = []

    for var in scfg.get('variables', []):
        if var not in df.columns:
            missing.append(var)
            continue

        series = df[var].dropna()
        if series.empty:
            issues.append(f"{var}: all NaN")
            continue

        bounds = scfg.get('range_checks', {}).get(var)
        if bounds:
            lo, hi = bounds
            n_out = ((series < lo) | (series > hi)).sum()
            pct   = 100 * n_out / len(series)
            if pct > 5:
                issues.append(f"{var}: {pct:.1f}% out of [{lo}, {hi}]")
            elif pct > 0:
                issues.append(f"{var}: {pct:.1f}% out of [{lo}, {hi}] (minor)")

    if missing:
        status = STATUS_FAIL
        detail = f"missing columns: {', '.join(missing)}"
    elif issues:
        status = STATUS_WARN
        detail = '; '.join(issues)
    else:
        status = STATUS_PASS
        detail = f"all {len(scfg.get('variables', []))} variables present and in range"

    print(f"[{status}] {label} — {check_name}: {detail}")
    return {'sensor': label, 'check': check_name, 'status': status, 'detail': detail}


# =============================================================================
# IMU sampling rate diagnostic (rich HTML table)
# =============================================================================

def check_imu_sampling_rate(
    df_imu:        Optional[pd.DataFrame],
    scfg:          dict,
    label:         str,
    config_change: Optional[str] = None,
) -> dict:
    """
    Detailed sampling rate diagnostic for the IMU stream.

    Computes the distribution of inter-sample intervals, classifies each
    bucket (nominal, jitter, missed beats, duplicates, gaps), and renders
    an HTML summary table with a bar chart column.

    Parameters
    ----------
    df_imu : DataFrame with DatetimeIndex
    scfg   : gpswaves.imu config dict (expected_interval_s, tolerance_s)
    label  : sensor label for the summary row
    config_change : optional string describing a known config change,
                    e.g. "09:24 — sample rate updated from 250 ms to 50 ms"
                    If provided, the data is split at that time and each
                    segment is shown separately.

    Returns
    -------
    dict suitable for the summary table
    """
    check_name = 'IMU sampling rate'

    if df_imu is None or df_imu.empty:
        print(f"[{STATUS_FAIL}] {label} — {check_name}: no data")
        return {'sensor': label, 'check': check_name,
                'status': STATUS_FAIL, 'detail': 'no data'}

    expected_s = scfg['expected_interval_s']
    tol_s      = scfg['tolerance_s']

    # If a config change time is given, split on it
    if config_change is not None:
        # Accept 'HH:MM' or full 'YYYY-MM-DD HH:MM'
        if len(str(config_change)) <= 5:
            date_str = df_imu.index[0].strftime('%Y-%m-%d')
            split_ts = pd.Timestamp(f"{date_str} {config_change}", tz='UTC')
        else:
            split_ts = pd.Timestamp(config_change, tz='UTC')
        segments = {
            f'Before {config_change}': df_imu[df_imu.index < split_ts],
            f'After {config_change}':  df_imu[df_imu.index >= split_ts],
        }
    else:
        segments = {'Full record': df_imu}

    all_statuses = []

    for seg_name, seg_df in segments.items():
        if seg_df.empty:
            continue
        status, detail = _render_imu_interval_table(
            seg_df, expected_s, tol_s, title=f"{label} — {seg_name}"
        )
        all_statuses.append(status)

    # If a config change was noted, judge only the final (post-change) segment
    # since that's the configuration that will be used on the mission.
    # Otherwise take the worst status across all segments.
    if config_change and len(all_statuses) > 1:
        final_status = all_statuses[-1]  # last segment = post-change
        detail = (f"expected {expected_s*1000:.0f} ms ({1/expected_s:.0f} Hz) "
                  f"— judged on post-change segment only")
    else:
        if STATUS_FAIL in all_statuses:
            final_status = STATUS_FAIL
        elif STATUS_WARN in all_statuses:
            final_status = STATUS_WARN
        else:
            final_status = STATUS_PASS
        detail = f"expected {expected_s*1000:.0f} ms ({1/expected_s:.0f} Hz)"
        if config_change:
            detail += f"; config change noted at {config_change}"

    return {'sensor': label, 'check': check_name,
            'status': final_status, 'detail': detail}


def _render_imu_interval_table(
    df:         pd.DataFrame,
    expected_s: float,
    tol_s:      float,
    title:      str,
) -> tuple[str, str]:
    """
    Compute interval distribution and render as an HTML table.
    Returns (status, detail_string).
    """
    dt_ns = df.index.to_series().diff().dropna().values.astype('int64')
    total = len(dt_ns)

    # Bucket counts
    unique_ns, counts = np.unique(dt_ns, return_counts=True)

    expected_ns = int(expected_s * 1e9)
    tol_ns      = int(tol_s * 1e9)

    def classify(ns):
        ms = ns / 1e6
        if ns == 0:                          return 'duplicate'
        if abs(ns - expected_ns) <= tol_ns:  return 'nominal'
        if ms < 1:                           return 'duplicate'
        if ms > 1000:                        return 'gap'
        return 'other'

    nominal_count = sum(c for ns, c in zip(unique_ns, counts)
                        if classify(ns) == 'nominal')
    pct_nominal   = 100 * nominal_count / total

    # Status
    if pct_nominal >= 99:
        status = STATUS_PASS
    elif pct_nominal >= 90:
        status = STATUS_WARN
    else:
        status = STATUS_FAIL

    # Build HTML table
    status_colors = {
        'nominal':   ('#ccffcc', '#2d6a2d'),
        'duplicate': ('#fff0cc', '#7a5c00'),
        'gap':       ('#ffcccc', '#8a2020'),
        'other':     ('#f0f0f0', '#555555'),
    }

    max_count = max(counts)

    rows_html = ''
    for ns, count in zip(unique_ns, counts):
        ms   = ns / 1e6
        hz   = f"{1000/ms:.1f} Hz" if ms > 0.5 else '—'
        pct  = 100 * count / total
        cls  = classify(ns)
        bg, fg = status_colors[cls]
        bar_w  = int(120 * count / max_count)

        rows_html += f"""
        <tr style="background:{bg}; color:{fg};">
          <td style="padding:5px 10px;">{ms:.1f} ms</td>
          <td style="padding:5px 10px;">{hz}</td>
          <td style="padding:5px 10px; text-align:right; font-variant-numeric:tabular-nums;">{count:,}</td>
          <td style="padding:5px 10px; text-align:right;">{('<0.01' if pct < 0.01 else f'{pct:.2f}') + '%'}</td>
          <td style="padding:5px 10px;">
            <div style="background:#ddd;border-radius:3px;height:8px;width:120px;display:inline-block;vertical-align:middle;">
              <div style="background:{fg};border-radius:3px;height:8px;width:{bar_w}px;"></div>
            </div>
          </td>
          <td style="padding:5px 10px; font-size:12px;">{cls}</td>
        </tr>"""

    html = f"""
    <div style="margin: 1rem 0; font-family: sans-serif; font-size: 14px;">
      <div style="font-weight:500; font-size:15px; margin-bottom:0.5rem;">{title}</div>
      <div style="display:flex; gap:16px; margin-bottom:0.75rem; font-size:13px;">
        <span>Total intervals: <strong>{total:,}</strong></span>
        <span>On-rate ({expected_s*1000:.0f} ms ± {tol_s*1000:.0f} ms):
          <strong style="color:{'#2d6a2d' if status==STATUS_PASS else '#8a2020'}">
            {pct_nominal:.2f}%
          </strong>
        </span>
        <span>Status: {status}</span>
      </div>
      <table style="border-collapse:collapse; width:100%;">
        <thead>
          <tr style="border-bottom:1px solid #ccc; font-size:12px; color:#666;">
            <th style="padding:5px 10px; text-align:left;">Interval</th>
            <th style="padding:5px 10px; text-align:left;">Rate</th>
            <th style="padding:5px 10px; text-align:right;">Count</th>
            <th style="padding:5px 10px; text-align:right;">Share</th>
            <th style="padding:5px 10px;"></th>
            <th style="padding:5px 10px; text-align:left;">Class</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """

    display(HTML(html))
    return status, f"{pct_nominal:.1f}% nominal"


# =============================================================================
# ADCP-specific checks
# =============================================================================

def check_adcp(ds, scfg: dict) -> list[dict]:
    """Run ADCP-specific diagnostics on the decoded xarray Dataset."""
    label   = scfg['label']
    results = []

    if ds is None:
        return [{'sensor': label, 'check': 'Data loaded', 'status': STATUS_FAIL,
                 'detail': 'no dataset'}]

    if 'time' in ds.coords:
        times     = pd.to_datetime(ds['time'].values)
        dt        = pd.Series(times).diff().dt.total_seconds().dropna()
        median_dt = dt.median()
        expected  = scfg['expected_interval_s']
        tol       = scfg['tolerance_s']
        status    = STATUS_PASS if abs(median_dt - expected) <= tol else STATUS_FAIL
        detail    = f"median Δt = {median_dt:.3f}s (expected {expected}s)"
        print(f"[{status}] {label} — Sampling rate: {detail}")
        results.append({'sensor': label, 'check': 'Sampling rate',
                        'status': status, 'detail': detail})

    if 'correlation' in ds:
        corr     = ds['correlation'].values
        pct_good = 100 * (corr >= scfg['min_correlation']).mean()
        status   = STATUS_PASS if pct_good >= 70 else STATUS_WARN if pct_good >= 50 else STATUS_FAIL
        detail   = f"{pct_good:.1f}% above correlation threshold ({scfg['min_correlation']})"
        print(f"[{status}] {label} — Beam correlation: {detail}")
        results.append({'sensor': label, 'check': 'Beam correlation',
                        'status': status, 'detail': detail})

    if 'velocity' in ds:
        vel     = ds['velocity'].values
        lo, hi  = scfg['velocity_range']
        pct_out = 100 * ((vel < lo) | (vel > hi)).mean()
        status  = STATUS_PASS if pct_out < 5 else STATUS_WARN
        detail  = f"{pct_out:.1f}% outside [{lo}, {hi}] m/s"
        print(f"[{status}] {label} — Velocity range (dockside): {detail}")
        results.append({'sensor': label, 'check': 'Velocity range (dockside)',
                        'status': status, 'detail': detail})

    return results


# =============================================================================
# GPSWaves checks
# =============================================================================

def check_gpswaves(df_imu: Optional[pd.DataFrame],
                   df_gps: Optional[pd.DataFrame],
                   scfg:   dict,
                   imu_config_change: Optional[str] = None) -> list[dict]:
    """
    Check GPSWaves IMU stream and GPS data.

    Parameters
    ----------
    imu_config_change : optional 'HH:MM' string for a known sample rate
                        change during the test (e.g. '09:24'). If given,
                        the IMU table is split at that time.
    """
    label   = scfg['label']
    results = []

    # IMU sampling rate (detailed table)
    icfg = scfg['imu']
    if df_imu is not None and not df_imu.empty:
        result = check_imu_sampling_rate(
            df_imu, icfg, label=label,
            config_change=imu_config_change,
        )
        results.append(result)
    else:
        results.append({'sensor': label, 'check': 'IMU sampling rate',
                        'status': STATUS_FAIL, 'detail': 'no IMU data'})

    # GPS position sanity
    if df_gps is not None and not df_gps.empty:
        n_fix   = len(df_gps)
        n_valid = (df_gps.get('llh_flags', pd.Series(dtype=int)) > 0).sum()
        hacc_median = df_gps['hacc'].median() if 'hacc' in df_gps.columns else float('nan')
        status  = STATUS_PASS if hacc_median < 10 else STATUS_WARN
        detail  = f"{n_fix} GPS fixes, median hacc = {hacc_median:.1f} m"
        print(f"[{status}] {label} — GPS position: {detail}")
        results.append({'sensor': label, 'check': 'GPS position',
                        'status': status, 'detail': detail})
    else:
        results.append({'sensor': label, 'check': 'GPS position',
                        'status': STATUS_WARN, 'detail': 'no GPS data'})

    return results


# =============================================================================
# Cross-sensor time alignment
# =============================================================================

def check_time_alignment(streams: dict[str, pd.DataFrame], xcfg: dict) -> dict:
    """Compare the start time of each sensor stream to a reference sensor."""
    check_name = 'Clock alignment'
    ref_key    = xcfg.get('reference_sensor', 'rbr')
    max_offset = xcfg.get('max_offset_s', 5.0)

    if ref_key not in streams:
        return {'sensor': 'Cross-sensor', 'check': check_name, 'status': STATUS_SKIP,
                'detail': f"reference sensor '{ref_key}' not loaded"}

    ref_start = streams[ref_key].index[0]
    offsets   = {}
    for name, df in streams.items():
        if name == ref_key or df is None or df.empty:
            continue
        offset_s = abs((df.index[0] - ref_start).total_seconds())
        offsets[name] = offset_s

    bad = {k: v for k, v in offsets.items() if v > max_offset}
    if bad:
        status = STATUS_WARN
        detail = "large start offsets: " + ", ".join(f"{k}={v:.1f}s" for k, v in bad.items())
    else:
        status = STATUS_PASS
        detail = f"all streams within {max_offset}s of reference ({ref_key})"

    print(f"[{status}] Cross-sensor — {check_name}: {detail}")
    return {'sensor': 'Cross-sensor', 'check': check_name, 'status': status, 'detail': detail}


# =============================================================================
# Plotting helpers
# =============================================================================

from src.plots import _add_phase_bar


def plot_timeseries(
    df:        'Optional[pd.DataFrame]',
    variables: list[str],
    title:     str,
    phases:    list[dict] | None = None,
    events:    list[dict] | None = None,
):
    """
    Standard multi-panel time series plot used for ALL sensors.
    Phase bar is drawn as an inset above the top panel (SPN1 style).
    Event markers are vertical dashed lines on all panels.
    """
    if df is None or df.empty:
        print(f"  [plot skipped — no data for {title}]")
        return

    vars_present = [v for v in variables if v in df.columns]
    if not vars_present:
        print(f"  [plot skipped — none of {variables} found in {title}]")
        return

    n = len(vars_present)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n),
                             sharex=True,
                             gridspec_kw={'hspace': 0.08})
    if n == 1:
        axes = [axes]

    reference_date = df.index[0]

    # Parse event timestamps once
    parsed_events = []
    if events:
        for ev in events:
            try:
                t_str = str(ev['time'])
                if len(t_str) <= 5:
                    t_ev = pd.Timestamp(
                        f"{reference_date.date()} {t_str}",
                        tz=reference_date.tzinfo,
                    )
                else:
                    t_ev = pd.Timestamp(t_str)
                    if t_ev.tzinfo is None:
                        t_ev = t_ev.tz_localize(reference_date.tzinfo or 'UTC')
                parsed_events.append({**ev, '_ts': t_ev})
            except Exception:
                pass

    # Data panels
    for ax, var in zip(axes, vars_present):
        ax.plot(df.index, df[var], lw=0.6, color='steelblue')
        ax.set_ylabel(var, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        for ev in parsed_events:
            ax.axvline(ev['_ts'], color=ev.get('color', 'gray'),
                       lw=0.8, alpha=0.6, ls='--')

    # Title on top panel
    axes[0].set_title(title, fontsize=10, fontweight='bold', pad=4)

    # Phase bar as inset above top panel — pass data range so phases align
    if phases:
        _test_period = (df.index[0], df.index[-1])
        _add_phase_bar(axes[0], phases, reference_date, test_period=_test_period)

    # Date formatting
    span = (df.index[-1] - df.index[0]).total_seconds()
    fmt = mdates.DateFormatter('%H:%M') if span < 86400           else mdates.DateFormatter('%b %d\n%H:%M')
    axes[-1].xaxis.set_major_formatter(fmt)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=0, ha='center')
    plt.tight_layout()
    plt.show()


def plot_data_coverage(streams: dict[str, pd.DataFrame], title: str = "Data coverage"):
    """Gantt-style plot showing the time extent of each sensor stream."""
    if not streams:
        print("[plot skipped — no streams loaded]")
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(streams) * 0.6)))
    colours = plt.cm.tab10.colors

    yticks, ylabels = [], []
    for i, (name, df) in enumerate(streams.items()):
        if df is None or df.empty:
            continue
        t0 = mdates.date2num(df.index[0].to_pydatetime())
        t1 = mdates.date2num(df.index[-1].to_pydatetime())
        ax.barh(i, t1 - t0, left=t0, height=0.6,
                color=colours[i % len(colours)], alpha=0.8)
        yticks.append(i)
        ylabels.append(name)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Summary table
# =============================================================================

def render_summary(results: list[dict], mission: str = ""):
    """
    Render the final summary table as a styled DataFrame.
    If results contain a 'session' key, inserts a bold header row
    between sessions so the overall summary is easy to scan.
    """
    if not results:
        print("No results to summarise.")
        return

    df = pd.DataFrame(results)

    n_pass = (df['status'] == STATUS_PASS).sum()
    n_warn = (df['status'] == STATUS_WARN).sum()
    n_fail = (df['status'] == STATUS_FAIL).sum()

    print(f"\n{'='*60}")
    print(f"  PREFLIGHT SUMMARY  —  {mission}")
    print(f"  ✅ {n_pass} passed   ⚠️  {n_warn} warnings   ❌ {n_fail} failed")
    print(f"{'='*60}\n")

    def colour_status(val):
        if val == STATUS_FAIL:  return 'background-color: #ffcccc'
        if val == STATUS_WARN:  return 'background-color: #fff3cc'
        if val == STATUS_PASS:  return 'background-color: #ccffcc'
        return ''

    # If multiple sessions, insert divider rows between them
    has_sessions = 'session' in df.columns and df['session'].nunique() > 1
    if has_sessions:
        cols = [c for c in ['session', 'sensor', 'check', 'status', 'detail'] if c in df.columns]
        df = df[cols]
        frames = []
        for session_name, group in df.groupby('session', sort=False):
            divider = pd.DataFrame([{
                'session': '',
                'sensor':  f'── {session_name} ──',
                'check':   '', 'status': '', 'detail': '',
            }])
            frames.append(divider)
            frames.append(group)
        df = pd.concat(frames, ignore_index=True)

    def style_row(row):
        styles = [''] * len(row)
        idx = row.index.tolist()
        # Bold divider rows
        if 'sensor' in idx and str(row['sensor']).startswith('──'):
            return [
                'font-weight: bold; background-color: #f0f0f0; '
                'border-top: 2px solid #aaa;'
            ] * len(row)
        # Colour status column
        if 'status' in idx:
            s = row['status']
            color = ('#ccffcc' if s == STATUS_PASS else
                     '#fff3cc' if s == STATUS_WARN else
                     '#ffcccc' if s == STATUS_FAIL else '')
            if 'status' in idx:
                styles[idx.index('status')] = f'background-color: {color}'
        return styles

    styled = df.style.apply(style_row, axis=1)
    try:
        display(styled)
    except ImportError:
        print(df.to_string(index=False))


# =============================================================================
# Session-aware data coverage plot
# =============================================================================

def plot_data_coverage_sessions(
    streams:  dict[str, 'pd.DataFrame'],
    sessions: list[dict],
    title:    str = "Data coverage",
    min_gap_h: float = 0.5,
):
    """
    Gantt-style coverage plot showing contiguous data blocks per sensor.
    Gaps > min_gap_h are shown as breaks in the bar.
    Session boundaries are shown as vertical shaded regions.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import pandas as pd
    import numpy as np

    fig, ax = plt.subplots(figsize=(14, max(3, len(streams) * 0.7)))
    colors = plt.cm.tab10.colors

    yticks, ylabels = [], []

    for i, (name, df) in enumerate(streams.items()):
        if df is None or (hasattr(df, 'empty') and df.empty):
            yticks.append(i)
            ylabels.append(name)
            continue

        # Get time index
        if isinstance(df, pd.DataFrame):
            idx = df.index
        else:
            idx = pd.DatetimeIndex(df.coords['time'].values)

        if len(idx) == 0:
            yticks.append(i)
            ylabels.append(name)
            continue

        # Find contiguous blocks separated by gaps > min_gap_h
        dt_h = pd.Series(idx).diff().dt.total_seconds().div(3600).fillna(0)
        breaks = np.where(dt_h > min_gap_h)[0]

        block_starts = [idx[0]] + [idx[b] for b in breaks]
        block_ends   = [idx[b-1] for b in breaks] + [idx[-1]]

        color = colors[i % len(colors)]
        for t0, t1 in zip(block_starts, block_ends):
            t0n = mdates.date2num(pd.Timestamp(t0).to_pydatetime())
            t1n = mdates.date2num(pd.Timestamp(t1).to_pydatetime())
            ax.barh(i, t1n - t0n, left=t0n, height=0.6,
                    color=color, alpha=0.85)

        yticks.append(i)
        ylabels.append(f"{name}  ({len(idx):,})")

    # Session boundary shading
    session_colors = ['#aed6f1', '#a9dfbf', '#f9e79f', '#f5cba7']
    for j, s in enumerate(sessions):
        t0n = mdates.date2num(s['start'].to_pydatetime())
        t1n = mdates.date2num(s['end'].to_pydatetime())
        ax.axvspan(t0n, t1n, alpha=0.08,
                   color=session_colors[j % len(session_colors)])
        ax.text(t0n, len(streams) - 0.1, f"  {s['name']}",
                fontsize='small', va='top', color='#555555')

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize='small')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title(title, fontsize='large', fontweight='medium')
    plt.tight_layout()
    plt.show()


# =============================================================================
# GPS quality check — context-aware (indoor vs outdoor)
# =============================================================================

def check_gps_quality(
    df_gps:  'pd.DataFrame',
    phases:  list[dict],
    label:   str = 'GPS',
) -> list[dict]:
    """
    Check GPS quality (num_sv, fix_type) in context of location phases.
    Indoor phases are expected to have poor GPS — flag only outdoor degradation.
    Returns list of summary dicts.
    """
    if df_gps is None or df_gps.empty:
        return []

    results = []

    # Split into indoor and outdoor periods based on phase labels
    outdoor_keywords = {'outdoor', 'outside', 'gantry', 'sky', 'direct sun', 'sun'}
    outdoor_mask = pd.Series(False, index=df_gps.index)

    if phases:
        ref = df_gps.index[0]
        for ph in phases:
            try:
                t_str = str(ph['start'])
                if len(t_str) <= 5:
                    t0 = pd.Timestamp(f"{ref.date()} {t_str}", tz=ref.tzinfo)
                else:
                    t0 = pd.Timestamp(t_str)
                    if t0.tzinfo is None:
                        t0 = t0.tz_localize(ref.tzinfo or 'UTC')
                t_str = str(ph['end'])
                if len(t_str) <= 5:
                    t1 = pd.Timestamp(f"{ref.date()} {t_str}", tz=ref.tzinfo)
                else:
                    t1 = pd.Timestamp(t_str)
                    if t1.tzinfo is None:
                        t1 = t1.tz_localize(ref.tzinfo or 'UTC')
                lbl = ph.get('label', '').lower()
                if any(kw in lbl for kw in outdoor_keywords):
                    outdoor_mask.loc[t0:t1] = True
            except Exception:
                pass

    n_total   = len(df_gps)
    n_outdoor = int(outdoor_mask.sum())
    n_indoor  = n_total - n_outdoor

    for col, thresh, name in [
        ('num_sv',   4, 'Satellites (num_sv)'),
        ('fix_type', 2, 'Fix type'),
    ]:
        if col not in df_gps.columns:
            continue

        # Overall
        n_bad_all = int((df_gps[col] < thresh).sum())

        # Split by context
        if n_outdoor > 0:
            df_out = df_gps[outdoor_mask]
            n_bad_out = int((df_out[col] < thresh).sum())
            pct_out   = 100 * n_bad_out / n_outdoor
        else:
            n_bad_out = n_bad_all
            pct_out   = 100 * n_bad_all / n_total if n_total else 0

        if n_indoor > 0:
            df_in   = df_gps[~outdoor_mask]
            n_bad_in  = int((df_in[col] < thresh).sum())
            pct_in    = 100 * n_bad_in / n_indoor
        else:
            n_bad_in = 0
            pct_in   = 0

        # Only flag outdoor degradation
        if pct_out > 20:
            status = STATUS_FAIL
        elif pct_out > 5:
            status = STATUS_WARN
        else:
            status = STATUS_PASS

        detail = (
            f"outdoor: {pct_out:.1f}% below threshold ({n_bad_out}/{n_outdoor})"
            + (f"; indoor: {pct_in:.1f}% (expected)" if n_indoor > 0 else "")
        )
        results.append({'sensor': label, 'check': name,
                        'status': status, 'detail': detail})

    return results