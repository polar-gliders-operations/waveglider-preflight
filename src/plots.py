"""
src/plots.py
------------
Sensor-specific diagnostic plots for the preflight notebook.

Each function produces a self-contained figure with:
  - Time series of the relevant variables
  - Event marker annotations (from the dockside report)
  - A stats panel with key diagnostic numbers

Event markers are passed as a list of dicts:
    [{'time': '08:49', 'label': 'moved outdoors', 'color': 'green'}, ...]
Times are assumed to match the data timezone (verified by whether markers
land on expected features).
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# =============================================================================
# Helpers
# =============================================================================

def _parse_event_times(events: list[dict], reference_date: pd.Timestamp) -> list[dict]:
    """
    Convert event times to full Timestamps.
    Accepts either:
      - 'HH:MM'              → prepends reference_date's date
      - 'YYYY-MM-DD HH:MM'   → parsed directly, tz applied
    """
    out = []
    for ev in events:
        try:
            t_str = str(ev['time'])
            if len(t_str) <= 5:  # HH:MM
                t = pd.Timestamp(
                    f"{reference_date.date()} {t_str}",
                    tz=reference_date.tzinfo,
                )
            else:  # full datetime string
                t = pd.Timestamp(t_str)
                if t.tzinfo is None:
                    t = t.tz_localize(reference_date.tzinfo or 'UTC')
            out.append({**ev, 'timestamp': t})
        except Exception:
            warnings.warn(f"Could not parse event time: {ev['time']!r}")
    return out


def _add_phase_bar(ax, phases: list[dict], reference_date: pd.Timestamp,
                   test_period=None):
    """
    Draw a horizontal phase bar just above the plot using an inset axes.
    Each phase is a coloured rectangle labelled with its name.
    """
    if not phases:
        return

    # Create a thin inset axes above the main plot (in figure fraction coords
    # is unreliable with constrained_layout; use a divider approach instead)
    ax_bar = ax.inset_axes([0, 1.01, 1, 0.045], transform=ax.transAxes)
    ax_bar.set_axis_off()

    # Determine the full time span to normalise rectangle widths
    if test_period:
        t0, t1 = test_period
    else:
        t0 = pd.Timestamp(f"{reference_date.date()} 00:00", tz=reference_date.tzinfo)
        t1 = pd.Timestamp(f"{reference_date.date()} 23:59", tz=reference_date.tzinfo)

    total_s = (t1 - t0).total_seconds()

    seen_labels = {}  # deduplicate legend entries

    def _parse_phase_time(t_str, reference_date):
        t_str = str(t_str)
        if len(t_str) <= 5:  # HH:MM
            t = pd.Timestamp(f"{reference_date.date()} {t_str}",
                             tz=reference_date.tzinfo)
        else:
            t = pd.Timestamp(t_str)
            if t.tzinfo is None:
                t = t.tz_localize(reference_date.tzinfo or 'UTC')
        return t

    for ph in phases:
        try:
            ph_start = _parse_phase_time(ph['start'], reference_date)
            ph_end   = _parse_phase_time(ph['end'],   reference_date)
        except Exception:
            continue

        x  = (ph_start - t0).total_seconds() / total_s
        w  = (ph_end   - ph_start).total_seconds() / total_s
        x  = max(0.0, x)
        w  = min(w, 1.0 - x)
        if w <= 0:
            continue

        color = ph.get('color', '#dddddd')
        rect  = mpatches.FancyBboxPatch(
            (x, 0), w, 1,
            boxstyle='square,pad=0',
            facecolor=color, edgecolor='white', lw=0.5,
            transform=ax_bar.transAxes, clip_on=False,
        )
        ax_bar.add_patch(rect)

        # Label if wide enough
        if w > 0.04:
            ax_bar.text(
                x + w / 2, 0.5, ph['label'],
                ha='center', va='center',
                fontsize='x-small', color='#333333',
                transform=ax_bar.transAxes, clip_on=False,
            )


def _filter_events(events: list[dict], skip_tags: list[str]) -> list[dict]:
    """Return events that do not carry any of the skip_tags."""
    if not skip_tags:
        return events
    return [e for e in events
            if not any(t in e.get('tags', []) for t in skip_tags)]


def _add_point_events(ax, events: list[dict], index: pd.DatetimeIndex):
    """Draw subtle vertical tick marks for point events (no vertical lines)."""
    if not events:
        return
    ymin, ymax = ax.get_ylim()
    for ev in events:
        t = ev['timestamp']
        if index[0] <= t <= index[-1]:
            color = ev.get('color', '#888888')
            # Short tick at top of plot only — less visual clutter than a full vline
            ax.axvline(t, color=color, lw=0.8, ls=':', alpha=0.5,
                       ymin=0.92, ymax=1.0)


def _stats_table(ax, rows: list[tuple], title: str = ""):
    """
    Render a clean stats table in a matplotlib axes (no spines, no ticks).
    rows: list of (label, value, status_char_or_None) or
                  (label, value, status_char_or_None, value_color_or_None)
    When value_color is provided it overrides the flag-derived color for the
    value column — used to match the line color in temperature comparison plots.
    """
    ax.set_axis_off()
    if title:
        ax.text(0, 1.0, title, fontsize='medium', fontweight='medium',
                va='top', transform=ax.transAxes)
        ax.plot([0, 1], [0.935, 0.935], color='#cccccc', lw=0.5,
                transform=ax.transAxes, clip_on=False)

    y = 0.88 if title else 1.0
    row_h = min(0.13, 0.88 / max(len(rows), 1))

    for row in rows:
        label, value, flag = row[0], row[1], row[2]
        line_color = row[3] if len(row) > 3 else None

        flag_color = '#2d6a2d' if flag == '✅' else '#8a2020' if flag == '❌' else \
                     '#7a5c00' if flag == '⚠️' else '#333333'
        value_color = line_color if line_color else flag_color

        ax.text(0.0,  y, label,      fontsize='medium', va='top',
                color='#555555', transform=ax.transAxes)
        ax.text(0.65, y, str(value), fontsize='medium', va='top',
                color=value_color, fontweight='medium', transform=ax.transAxes)
        if flag and flag not in ('', None):
            ax.text(0.97, y, flag, fontsize='medium', va='top',
                    transform=ax.transAxes)
        y -= row_h


def _style_time_axis(ax, test_period):
    """Apply standard time-axis formatting and optional x-limits."""
    if test_period:
        ax.set_xlim(test_period[0], test_period[1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.tick_params(labelsize='medium')
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_xlabel('')


# =============================================================================
# SPN1 Pyranometer (shortwave)
# =============================================================================

def plot_spn1(
    df:               pd.DataFrame,
    scfg:             dict,
    test_period:      Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    events:           Optional[list[dict]] = None,
    phases:           Optional[list[dict]] = None,
    outdoor_window:   Optional[tuple[str, str]] = None,
    saturation_mv:    Optional[float] = None,
) -> dict:
    """
    Diagnostic plot for the SPN1 pyranometer (shortwave radiation).

    Parameters
    ----------
    df            : DataFrame with DatetimeIndex, columns from scfg['variables']
    scfg          : cr6.spn1 config dict
    test_period   : (start, end) Timestamps — sets x-axis limits
    events        : list of event marker dicts {'time': 'HH:MM', 'label': ..., 'color': ...}
    outdoor_window: ('HH:MM', 'HH:MM') — start/end of outdoor period for mean calculation
    saturation_mv : known saturation voltage; if None uses 0.99 * max observed
    """
    col_total   = next((c for c in df.columns if 'total'   in c.lower()), None)
    col_diffuse = next((c for c in df.columns if 'diffuse' in c.lower()), None)

    if col_total is None or col_diffuse is None:
        warnings.warn("plot_spn1: could not identify total/diffuse columns")
        return {}

    total   = df[col_total].dropna()
    diffuse = df[col_diffuse].dropna()

    if test_period:
        start, end = test_period
        df_plot = df.loc[start:end]
    else:
        df_plot = df

    # --- Stats ---
    n_actual   = len(df)
    n_expected = None
    if test_period:
        duration_s = (test_period[1] - test_period[0]).total_seconds()
        n_expected = int(round(duration_s / scfg['expected_interval_s']))

    pct_diff_gt_total = 100 * (
        (df_plot[col_total] - df_plot[col_diffuse]) < 0
    ).sum() / max(len(df_plot), 1)

    max_total   = total.max()
    max_total_t = total.idxmax().strftime('%H:%M') if not total.empty else '—'
    max_diff    = diffuse.max()
    max_diff_t  = diffuse.idxmax().strftime('%H:%M') if not diffuse.empty else '—'

    mean_outdoor = None
    t_out_s = t_out_e = None
    if outdoor_window and test_period:
        date_str = test_period[0].strftime('%Y-%m-%d')
        tz       = test_period[0].tzinfo
        t_out_s  = pd.Timestamp(f"{date_str} {outdoor_window[0]}", tz=tz)
        t_out_e  = pd.Timestamp(f"{date_str} {outdoor_window[1]}", tz=tz)
        outdoor_total = df_plot.loc[t_out_s:t_out_e, col_total].dropna()
        if not outdoor_total.empty:
            mean_outdoor = outdoor_total.mean()

    noise_floor = None
    if t_out_s is not None:
        indoor_vals = df_plot.loc[df_plot.index < t_out_s, col_total].dropna()
        if not indoor_vals.empty:
            noise_floor = indoor_vals.median()

    sat_thresh  = (saturation_mv * 0.99) if saturation_mv else (total.max() * 0.99)
    n_saturated = int((total >= sat_thresh).sum())

    pct_complete     = 100 * n_actual / n_expected if n_expected else None
    completeness_flag = ('✅' if (pct_complete or 0) >= 95
                         else '⚠️' if (pct_complete or 0) >= 80 else '❌') if n_expected else '—'
    diff_gt_flag  = '✅' if pct_diff_gt_total < 1 else '⚠️' if pct_diff_gt_total < 5 else '❌'
    sat_flag      = '✅' if n_saturated == 0 else '⚠️'
    noise_flag    = '✅' if (noise_floor is not None and noise_floor < 50) else \
                    '⚠️' if noise_floor is not None else '—'

    stats = {
        'pct_diffuse_gt_total': pct_diff_gt_total,
        'max_total_mv': max_total, 'max_diffuse_mv': max_diff,
        'mean_outdoor_mv': mean_outdoor, 'noise_floor_mv': noise_floor,
        'n_saturated': n_saturated, 'n_actual': n_actual, 'n_expected': n_expected,
    }

    # --- Figure ---
    fig = plt.figure(figsize=(14, 4.5), constrained_layout=True)
    gs  = GridSpec(1, 5, figure=fig)
    ax_plot  = fig.add_subplot(gs[0, :4])
    ax_stats = fig.add_subplot(gs[0, 4])

    colors = {'total': '#E07B39', 'diffuse': '#3A7ABF'}
    ax_plot.plot(df_plot.index, df_plot[col_total],
                 color=colors['total'],   lw=1.8, label=col_total)
    ax_plot.plot(df_plot.index, df_plot[col_diffuse],
                 color=colors['diffuse'], lw=1.8, label=col_diffuse)

    if t_out_s is not None and t_out_e is not None:
        ax_plot.axvspan(t_out_s, t_out_e, alpha=0.06, color='gold', label='outdoor period')

    _style_time_axis(ax_plot, test_period)
    ax_plot.set_ylabel('mV', fontsize='large')
    ax_plot.legend(fontsize='medium', loc='upper left')
    ax_plot.set_title(
        f"{scfg['label']} — {col_diffuse}, {col_total}",
        fontsize='large', fontweight='medium',
    )

    if not df_plot.empty:
        ref = df_plot.index[0]
        if phases:
            _add_phase_bar(ax_plot, phases, ref, test_period)
        if events:
            filtered = _filter_events(events, skip_tags=['imu'])
            parsed   = _parse_event_times(filtered, ref)
            _add_point_events(ax_plot, parsed, df_plot.index)

    table_rows = [
        ('Samples',
         f"{n_actual:,}" + (f" / {n_expected:,}" if n_expected else ''),
         completeness_flag),
        ('Diffuse > Total',   f"{pct_diff_gt_total:.2f}%",           diff_gt_flag),
        ('Max total',         f"{max_total:.0f} mV @ {max_total_t}", None),
        ('Max diffuse',       f"{max_diff:.0f} mV @ {max_diff_t}",   None),
        ('Mean (outdoors)',
         f"{mean_outdoor:.0f} mV" if mean_outdoor is not None else '—', None),
        ('Noise floor',
         f"{noise_floor:.0f} mV"  if noise_floor  is not None else '—', noise_flag),
        ('Saturation events', str(n_saturated), sat_flag),
    ]
    _stats_table(ax_stats, table_rows, title='Diagnostics')

    plt.show()
    return stats


# =============================================================================
# Derived variable decoders
# =============================================================================

def decode_gill_temperature(df: pd.DataFrame, source_col: str = 'Speed_sound') -> pd.Series:
    """
    Derive sonic virtual temperature from Gill anemometer output.

    The CR6 stores the Gill's Speed_sound column as sonic virtual temperature
    in Kelvin (not speed of sound in m/s):

        T (°C) = Speed_sound (K) - 273.15

    Sonic virtual temperature is slightly higher than true air temperature in
    humid conditions but agrees with WXT air temperature to ~0.5°C typically.
    """
    if source_col not in df.columns:
        warnings.warn(f"decode_gill_temperature: column '{source_col}' not found")
        return pd.Series(dtype=float, name='T_acoustic')
    return (df[source_col] - 273.15).rename('T_acoustic')


def decode_cgr4_body_temperature(
    df:         pd.DataFrame,
    source_col: str   = 'CGR4_Raw_Vtherm',
    r_fixed:    float = 1000.0,  # Ω — confirmed by technician (Matthew Pheiffer)
    beta:       float = 3297.0,  # K — derived from calibration cert anchor points
    r_inf:      float = 0.0158,  # Ω — pre-exponential factor
) -> pd.Series:
    """
    Derive CGR4 body temperature from YSI 44031 thermistor voltage.

    Thermistor resistance (Matthew Pheiffer, technician):
        Rs = Rf * X / (1 - X)     [Rf = 1kΩ, X = Vtherm in V]

    Temperature from NTC beta model:
        T (K) = beta / ln(Rs / R_inf)

    Beta coefficients anchored to:
      - Calibration cert ambient: Vtherm ≈ 0.59 V @ 15.5°C
      - Outdoor warm period:      Vtherm ≈ 0.50 V @ 25.0°C
    → B = 3297 K, R_inf = 0.0158 Ω

    Note: the CR6 computes T_body_C directly using the full Steinhart-Hart
    equation. Prefer that column if available.
    """
    if source_col not in df.columns:
        warnings.warn(f"decode_cgr4_body_temperature: column '{source_col}' not found")
        return pd.Series(dtype=float, name='T_body')
    X  = df[source_col].clip(0.001, 0.999)
    Rs = r_fixed * X / (1.0 - X)
    T_K = beta / np.log(Rs / r_inf)
    return (T_K - 273.15).rename('T_body')


def decode_cgr4_longwave(
    df:          pd.DataFrame,
    mvolt_col:   str   = 'CGR4_Raw_mVolt',
    vtherm_col:  str   = 'CGR4_Raw_Vtherm',
    sensitivity: float = 13.25e-6,  # V/W/m² — calibration cert 032371300054
) -> pd.Series:
    """
    Compute downwelling longwave radiation from CGR4 raw outputs.

    Formula (from calibration certificate):
        Ld = U_emf / S + σ × Tb⁴

    where U_emf = CGR4_Raw_mVolt × 1e-3 (V), S = 13.25e-6 V/W/m², Tb in K.

    Prefer CGR4_Rad from the CR6 if available — it uses the full Steinhart-Hart.
    This function is for verification or when the derived column is absent.
    """
    SIGMA = 5.670374419e-8
    if mvolt_col not in df.columns:
        warnings.warn(f"decode_cgr4_longwave: column '{mvolt_col}' not found")
        return pd.Series(dtype=float, name='LW_down_Wm2')
    T_K   = decode_cgr4_body_temperature(df, source_col=vtherm_col) + 273.15
    U_emf = df[mvolt_col] * 1e-3
    return (U_emf / sensitivity + SIGMA * T_K**4).rename('LW_down_Wm2')


# =============================================================================
# Cross-sensor temperature comparison plot
# =============================================================================

def _plot_temp_diff(
    ax_plot, ax_stats, series, label_a, label_b, test_period, phases=None,
):
    """
    Plot the difference (a - b) between two named temperature series.
    Fills above/below zero to make sign immediately obvious.
    """
    ser_a = next((s['series'] for s in series if s['label'] == label_a), None)
    ser_b = next((s['series'] for s in series if s['label'] == label_b), None)

    if ser_a is None or ser_b is None:
        ax_plot.set_visible(False)
        ax_stats.set_visible(False)
        return

    color_a = _WONG_PALETTE.get(label_a, '#333333')

    # Resample both to 1-min and align
    r_a = ser_a.resample('1min').mean()
    r_b = ser_b.resample('1min').mean()
    common = r_a.index.intersection(r_b.index)
    diff = (r_a.loc[common] - r_b.loc[common]).dropna()

    if diff.empty:
        ax_plot.set_visible(False)
        ax_stats.set_visible(False)
        return

    ax_plot.plot(diff.index, diff.values, color=color_a, lw=1.4)
    ax_plot.fill_between(diff.index, diff.values, 0,
                         where=diff.values >= 0,
                         color=color_a, alpha=0.15)
    ax_plot.fill_between(diff.index, diff.values, 0,
                         where=diff.values < 0,
                         color='#56B4E9', alpha=0.15)
    ax_plot.axhline(0, color='#aaaaaa', lw=0.8, ls='-')

    _style_time_axis(ax_plot, test_period)
    ax_plot.set_ylabel(f'Δ °C', fontsize='medium')
    short_a = label_a.split('(')[0].strip()
    short_b = label_b.split('(')[0].strip()
    ax_plot.set_title(f'{short_a} − {short_b}', fontsize='medium',
                      fontweight='medium')

    if phases:
        ref = diff.index[0]
        _add_phase_bar(ax_plot, phases, ref, test_period)

    # Stats panel
    mean_diff = diff.mean()
    std_diff  = diff.std()
    max_diff_val  = diff.max()
    min_diff_val  = diff.min()

    table_rows = [
        ('Mean offset',  f'{mean_diff:+.2f} °C', None),
        ('Std',          f'{std_diff:.2f} °C',   None),
        ('Max',          f'{max_diff_val:+.2f} °C', None),
        ('Min',          f'{min_diff_val:+.2f} °C', None),
    ]
    _stats_table(ax_stats, table_rows, title=f'Δ {short_a}−{short_b}')



def plot_temperature_comparison(
    dataframes:  dict[str, pd.DataFrame],
    cfg:         dict,
    test_period: Optional[tuple] = None,
    events:      Optional[list]  = None,
    phases:      Optional[list]  = None,
) -> dict:
    """
    Cross-sensor temperature comparison.

    Air sensors (Gill, WXT, CGR4 body) are all plotted on a single axis.
    Water sensors (RBR, CODA) are plotted on a separate axis below.
    A stats panel shows per-sensor mean ± std and pairwise offsets.
    """
    tcfg = cfg.get('cross_sensor', {}).get('temperature_comparison', {})
    if not tcfg.get('enabled', True):
        print("[SKIP] cross-sensor temperature comparison disabled in config")
        return {}

    # Decode derived temperatures
    dfs = dict(dataframes)
    if 'gill' in dfs and dfs['gill'] is not None:
        dfs['gill'] = dfs['gill'].copy()
        dfs['gill']['T_acoustic'] = decode_gill_temperature(dfs['gill'])
    if 'cgr4' in dfs and dfs['cgr4'] is not None:
        dfs['cgr4'] = dfs['cgr4'].copy()
        dfs['cgr4']['T_body'] = decode_cgr4_body_temperature(dfs['cgr4'])

    # Clip to test period
    if test_period:
        start, end = test_period
        dfs = {k: (v.loc[start:end] if v is not None else None)
               for k, v in dfs.items()}

    # Collect all temperature series — air and water on the same axis
    # (dockside: everything is in air, sea trials handled separately)
    all_series = (
        _collect_temp_series(dfs, tcfg.get('air_sensors',   [])) +
        _collect_temp_series(dfs, tcfg.get('water_sensors', []))
    )

    if not all_series:
        print("[SKIP] no temperature series available")
        return {}

    # Use the most lenient threshold for the combined set
    max_diff = max(
        tcfg.get('max_air_disagreement_degc',   2.0),
        tcfg.get('max_water_disagreement_degc', 0.5),
    )

    # Two-row layout: main panel + half-height diff panel below
    # GridSpec rows weighted 2:1; 5 columns (4 plot + 1 stats)
    fig = plt.figure(constrained_layout=True, figsize=(14, 6))
    gs  = GridSpec(2, 5, figure=fig,
                   height_ratios=[2, 1],
                   hspace=0.08)
    ax_plot  = fig.add_subplot(gs[0, :4])
    ax_stats = fig.add_subplot(gs[0, 4])
    ax_diff  = fig.add_subplot(gs[1, :4], sharex=ax_plot)
    ax_diff_stats = fig.add_subplot(gs[1, 4])

    result = _plot_temp_panel(
        ax_plot, ax_stats,
        series      = all_series,
        title       = 'Temperature — all sensors',
        test_period = test_period,
        events      = events,
        max_diff    = max_diff,
        phases      = phases,
        skip_tags   = ['imu'],
    )

    # --- Gill vs WXT difference panel ---
    _plot_temp_diff(
        ax_diff, ax_diff_stats,
        series      = all_series,
        label_a     = 'Gill (acoustic)',
        label_b     = 'WXT (air)',
        test_period = test_period,
        phases      = phases,
    )

    # Hide x tick labels on main panel since axes are shared
    plt.setp(ax_plot.get_xticklabels(), visible=False)
    ax_plot.set_xlabel('')

    plt.show()
    return {'all': result}


def _collect_temp_series(dfs, sensors):
    out = []
    for s in sensors:
        df = dfs.get(s['sensor'])
        if df is not None and s['variable'] in df.columns:
            series = df[s['variable']].dropna()
            if not series.empty:
                out.append({'label': s.get('label', s['variable']),
                            'series': series, 'sensor': s['sensor']})
    return out


# Wong colorblind-safe palette assigned per sensor label.
# Fall back to cycling through the palette for unlisted sensors.
_WONG_PALETTE = {
    'Gill (acoustic)': '#E69F00',   # orange
    'WXT (air)':       '#56B4E9',   # sky blue
    'CGR4 (body)':     '#000000',   # black — blackbody radiometer
    'RBR (water)':     '#D62728',   # red — RBR mnemonic
    'CODA (water)':    '#CC79A7',   # purple
    'ADCP (water)':    '#0072B2',   # blue
}
_WONG_CYCLE = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7',
               '#0072B2', '#F0E442']

# Water sensors shown dashed to distinguish from air sensors
_WATER_SENSORS = {'rbr', 'coda'}


def _sensor_style(s: dict, idx: int) -> tuple[str, str]:
    """Return (color, linestyle) for a series dict."""
    color = _WONG_PALETTE.get(s['label'],
                               _WONG_CYCLE[idx % len(_WONG_CYCLE)])
    ls    = '--' if s.get('sensor', '') in _WATER_SENSORS else '-'
    return color, ls


def _plot_temp_panel(
    ax_plot, ax_stats, series, title, test_period, events, max_diff,
    phases=None, skip_tags=None,
) -> dict:
    """All temperature series on one axis, stats panel on the right."""

    # Build color map so stats table matches line colors
    color_map = {}
    for i, s in enumerate(series):
        color, ls = _sensor_style(s, i)
        color_map[s['label']] = color
        ser = s['series']
        # Downsample high-rate sensors (Gill 20 Hz → 1 min) for readability
        if ser.index.to_series().diff().dt.total_seconds().median() < 1:
            ser = ser.resample('1min').mean()
        ax_plot.plot(ser.index, ser.values,
                     color=color, lw=1.8, ls=ls, label=s['label'])

    _style_time_axis(ax_plot, test_period)
    ax_plot.set_ylabel('°C', fontsize='large')
    ax_plot.legend(fontsize='medium', loc='upper left')
    ax_plot.set_title(title, fontsize='large', fontweight='medium')

    if series:
        ref = series[0]['series'].index[0]
        if phases:
            _add_phase_bar(ax_plot, phases, ref, test_period)
        if events:
            filtered = _filter_events(events, skip_tags=skip_tags or [])
            parsed   = _parse_event_times(filtered, ref)
            _add_point_events(ax_plot, parsed, series[0]['series'].index)

    # Stats: mean ± std per sensor (colored), then pairwise offsets
    table_rows = []
    for s in series:
        ser = s['series']
        table_rows.append((
            s['label'],
            f"{ser.mean():.2f} ± {ser.std():.2f} °C",
            None,
            color_map.get(s['label'], '#333333'),   # line color for value
        ))

    offsets = []
    if len(series) >= 2:
        table_rows.append(('', '', None, None))
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                s1, s2 = series[i], series[j]
                r1 = s1['series'].resample('1min').mean().dropna()
                r2 = s2['series'].resample('1min').mean().dropna()
                common = r1.index.intersection(r2.index)
                if len(common) > 5:
                    mean_offset = (r1.loc[common] - r2.loc[common]).mean()
                    offsets.append(abs(mean_offset))
                    flag = ('✅' if abs(mean_offset) <= max_diff
                            else '⚠️' if abs(mean_offset) <= max_diff * 2
                            else '❌')
                    n1 = s1['label'].split('(')[0].strip()
                    n2 = s2['label'].split('(')[0].strip()
                    table_rows.append((f"{n1} vs {n2}",
                                       f"Δ = {mean_offset:+.2f} °C",
                                       flag, None))

    overall      = max(offsets) if offsets else 0
    overall_flag = ('✅' if overall <= max_diff
                    else '⚠️' if overall <= max_diff * 2 else '❌')

    _stats_table(ax_stats, table_rows, title='Diagnostics')

    return {'max_offset_degc': overall, 'status': overall_flag,
            'means': {s['label']: s['series'].mean() for s in series}}