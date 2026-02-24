# Sanash Bus Simulation

SimPy agent-based simulation modelling passenger boarding decisions under three information scenarios.

## What It Models

- **Route:** Single bus route with 8 stops
- **Buses:** 60-passenger capacity, Poisson headways (mean 10 min)
- **Passengers:** Arrive via Poisson process (3 pax/min/stop)
- **Boarding decision:** Binary logit (MNL) using configurable β coefficients

## Scenarios

| Scenario | Description |
|----------|-------------|
| `no_info` | Passengers board first available bus (no crowding awareness) |
| `perfect_info` | Passengers see exact occupancy, decide via MNL model |
| `imperfect_info` | Passengers see occupancy with ±10% uniform noise |

## Key Output Metrics

- **avg_wait_time** — Mean wait time per boarded passenger (min)
- **load_factor_cv** — Coefficient of variation of bus load factors (spread equity)
- **max_occupancy** — Peak occupancy ratio observed
- **total_boarded / total_refused** — Service throughput

## Running the Simulation

```bash
pip install -r simulation/requirements.txt

# Run all scenarios, 100 reps each:
python simulation/bus_simulation.py

# Quick test (10 reps):
python simulation/bus_simulation.py --replications 10

# Single scenario:
python simulation/bus_simulation.py --scenario perfect_info --replications 50
```

## Running Post-Simulation Analysis

```bash
python simulation/analysis.py
# or specify input:
python simulation/analysis.py --input simulation/output/simulation_results.csv
```

## Configuration

Edit `simulation/config.py` to change:
- `NUM_STOPS`, `BUS_CAPACITY`, `MEAN_HEADWAY`
- `BETA_*` coefficients (calibrate from survey DCE results)
- `NOISE_LEVEL` for imperfect information scenario
- `NUM_REPLICATIONS`, `SIM_DURATION`

## Output Files

Results saved to `simulation/output/`:
- `simulation_results.csv` — Raw replication results
- `wait_times_boxplot.png`
- `load_factor_cv.png`
- `occupancy_timeseries.png`
- `sensitivity_tornado.png`
- `anova_results.txt`
