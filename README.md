# waveglider-preflight

Pre-mission and dockside diagnostic tooling for the Wave Glider sensor suite.

Run before each deployment to verify that all sensors are present, sampling at
the expected rates, and returning values within sensible ranges. Produces a
colour-coded summary table (✅ / ⚠️ / ❌) for every check across every session.

---

## What it does

| Step | What it does |
|---|---|
| Parse & cache | Reads all raw sensor files, rejects corrupt timestamps, saves one NetCDF per sensor |
| Data coverage | Gantt plot showing contiguous data blocks per sensor across all sessions |
| Sampling rate | Checks median interval against expected rate per sensor |
| Variable ranges | Flags values outside physically sensible bounds |
| SPN1 pyranometer | Dedicated shortwave plot with outdoor shading, noise floor, saturation events |
| Temperature comparison | All temperature sensors overlaid (Gill acoustic, WXT, CGR4, RBR, CODA, ADCP) |
| Wave spectrogram | Log energy spectrum vs frequency vs time, Hs from integral, CHECK quality factor |
| IMU shake test | Full 20 Hz accelerometer zoom around motion-tagged events |
| Summary | One row per check, grouped by session, colour-coded |

---

## Quick start

```bash
git clone https://github.com/polar-gliders-operations/waveglider-preflight.git
cd waveglider-preflight
conda env create -f environment.yml
conda activate waveglider-preflight
```

1. Edit `config/mission_config.yml` — set `data_path`, `valid_data_range`, and define `sessions`
2. Run `notebooks/parse_and_cache.ipynb` once to build the cache
3. Run `notebooks/preflight_check.ipynb` for diagnostics — re-run as many times as needed

```bash
jupyter lab
```

---

## Workflow

### Two-notebook design

**`parse_and_cache.ipynb`** — run once per dataset:
- Parses all raw files (CR6 TOB3/TOA5, ADCP `.ENS` binary, GPSWaves `raw.log`, `waves.log`, `spectrum.log`)
- Rejects timestamps outside `valid_data_range` (catches corrupt records like year 2113)
- Saves one NetCDF per sensor to `<data_path>/cache/`
- Cell 5 auto-detects session boundaries from the data

**`preflight_check.ipynb`** — loads from cache (fast), iterates over sessions:
- Plots every sensor present; skips sensors absent — no configuration needed
- Phase bar and event markers on every plot
- Per-session and overall summary tables

### Cache layout

```
<data_path>/cache/
├── rbr.nc       wxt.nc      gill.nc
├── spn1.nc      cgr4.nc     avg.nc
├── adcp_beam.nc imu.nc      gps.nc
├── waves.nc     spectra.nc
```

---

## Sensor suite (WHIRLS 2026)

| Sensor | Source | Rate | Cache |
|---|---|---|---|
| RBR CTD + CODA O₂ + Cyclops Chl-a | CR6 TOB3/TOA5 | 1 Hz | `rbr.nc` |
| Vaisala WXT536 | CR6 TOB3/TOA5 | 1 Hz | `wxt.nc` |
| Gill Sonic Anemometer | CR6 TOB3/TOA5 | 20 Hz | `gill.nc` |
| SPN1 Pyranometer (shortwave) | CR6 TOB3/TOA5 | 30 s avg | `spn1.nc` |
| CGR4 Pyrgeometer (longwave) | CR6 TOB3/TOA5 | 30 s avg | `cgr4.nc` |
| RDI WorkHorse ADCP | `.ENS` binary | 1 Hz | `adcp_beam.nc` |
| MicroStrain IMU (GX4-45/GX5-45) | GPSWaves `raw.log` | 20 Hz | `imu.nc` |
| GPSWaves GPS fixes | GPSWaves `raw.log` | 4 Hz | `gps.nc` |
| Wave bulk parameters (Hs, Ta, Tp, Dp) | GPSWaves `waves.log` | 30 min | `waves.nc` |
| Directional wave spectra | GPSWaves `spectrum.log` | 30 min | `spectra.nc` |

---

## Configuration

### Sessions

Each session has a name, start/end, phases (coloured background bands), and
events (vertical markers). Sensors present = plotted; sensors absent = skipped.

```yaml
sessions:
  - name: "Dockside test 1"
    start: "2026-05-21 08:00"
    end:   "2026-05-21 15:00"
    phases:
      - {label: "indoors",  start: "08:00", end: "09:00", color: "#cccccc"}
      - {label: "outdoors", start: "09:00", end: "15:00", color: "#aed6f1"}
    events:
      - {time: "08:25", label: "sensors ON", color: "green", tags: ["sensor"]}
      - {time: "09:00", label: "moved outdoors", color: "green", tags: ["location"]}
```

For sessions spanning multiple days use `YYYY-MM-DD HH:MM` for phases and events.

### `valid_data_range`

Sanity filter applied at parse time. Timestamps outside this window are rejected
before caching. Set to the deployment year — independent of sessions, so
auto-detection (cell 5) always runs on clean data.

### Event tags

| Tag | Used for |
|---|---|
| `sensor` | Power-on/off events |
| `location` | Physical moves — derives outdoor window for SPN1 |
| `imu` | IMU config changes — splits sampling rate table at this time |
| `motion` | Shake/motion tests — triggers high-rate IMU zoom plot |
| `radiation` | Shadow/sun events |

---

## Repository structure

```
waveglider-preflight/
├── config/
│   └── mission_config.yml      all settings, sessions, phases, events
├── notebooks/
│   ├── parse_and_cache.ipynb   parse raw → NetCDF (run once per dataset)
│   └── preflight_check.ipynb   diagnostics (loads from cache, iterates sessions)
├── src/
│   ├── loaders.py              CR6 TOB3/TOA5 parser, apply_test_period
│   ├── adcp.py                 RDI WorkHorse ENS binary parser → xr.Dataset
│   ├── gpswaves.py             MIP protocol decoder (IMU set 0x80, GPS set 0x81)
│   ├── waves.py                waves.log + spectrum.log parsers
│   ├── diagnostics.py          check_* functions, plot_timeseries, render_summary
│   └── plots.py                plot_spn1, plot_temperature_comparison, _add_phase_bar
├── environment.yml
└── README.md
```

---

## Relationship to other repos

The decoders in `src/` are self-contained (Option A — no external dependencies
beyond the conda environment). The intention is to migrate towards installable
packages (Option B) once the processing repos stabilise:

- ADCP: [polar-gliders-operations/waveglider-adcp-processing](https://github.com/polar-gliders-operations/waveglider-adcp-processing)
- GPSWaves / XDATA: [polar-gliders-operations/waveglider-xdata](https://github.com/polar-gliders-operations/waveglider-xdata)

When those repos have a `pyproject.toml`, update `environment.yml` to install them as:

```yaml
  - pip:
    - git+https://github.com/polar-gliders-operations/waveglider-adcp-processing
    - git+https://github.com/polar-gliders-operations/waveglider-xdata
```

---

## Notes on specific sensors

**ADCP** — the binary parser does not use pycurrents. It reads `.ENS` files at
the byte level to handle the non-standard frame alignment found in Wave Glider
deployments.

**GPS quality** — `check_gps_quality()` is context-aware: indoor phases
(office, gantry, etc.) are expected to have poor GPS and are excluded from the
pass/fail threshold. Only outdoor degradation triggers a warning or failure.

**Wave data** — `9999` in `waves.log` and `spectrum.log` indicates invalid
bursts (insufficient GPS samples during the processing window). These are
retained as NaN rows with `valid=False` so gaps are visible in plots. The CHECK
variable in the spectra should be ≈1 for good deep-water data; large deviations
indicate poor vehicle response to swell or sensor issues.

---

## Authors

Johan Edholm, University of Gothenburg (UGOT)

Developed with assistance from Claude Sonnet 4.6 (Anthropic) — pair programming
across architecture design, binary protocol decoding, NetCDF serialisation, and
diagnostic plot implementation.

---

## References

Herbers, T.H.C., et al. (2012). Observing Ocean Surface Waves with GPS-Tracked
Buoys. *Journal of Atmospheric and Oceanic Technology*, 29, 944–959.