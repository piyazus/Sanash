"""
Sanash Bus Occupancy SimPy Simulation
======================================

Agent-based simulation of bus route passenger dynamics under three
information scenarios:

  - no_info:       Passengers board regardless of crowding
  - perfect_info:  Passengers see exact occupancy, decide via MNL model
  - imperfect_info: Passengers see occupancy with ±10% uniform noise

Usage:
    python simulation/bus_simulation.py
    python simulation/bus_simulation.py --replications 50 --output simulation/output/
    python simulation/bus_simulation.py --scenario perfect_info --replications 10

Output:
    CSV file at simulation/output/simulation_results.csv with one row per
    (scenario, replication) combination.
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import numpy as np
import pandas as pd

try:
    import simpy
except ImportError:
    print("SimPy not installed. Run: pip install simpy")
    sys.exit(1)

# Import config (handle both direct execution and module import)
try:
    from simulation import config as cfg
except ImportError:
    try:
        import config as cfg  # type: ignore
    except ImportError:
        # Inline defaults if config unavailable
        class cfg:  # type: ignore
            NUM_STOPS = 8
            BUS_CAPACITY = 60
            MEAN_HEADWAY = 10
            PASSENGER_ARRIVAL_RATE = 3
            BETA_WAIT = -0.15
            BETA_CROWDING_PACKED = -1.2
            BETA_CROWDING_STANDING = -0.6
            BETA_PEAK = -0.3
            NOISE_LEVEL = 0.10
            NUM_REPLICATIONS = 100
            SIM_DURATION = 240
            WARM_UP_PERIOD = 30
            PEAK_HOURS = [(0, 60), (180, 240)]
            RANDOM_SEED = 42
            OUTPUT_DIR = "simulation/output"
            RESULTS_FILENAME = "simulation_results.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCENARIOS = ["no_info", "perfect_info", "imperfect_info"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PassengerRecord:
    """Records the journey of a single simulated passenger."""
    passenger_id: int
    stop_id: int
    arrival_time: float
    board_time: float = 0.0
    alight_time: float = 0.0
    wait_time: float = 0.0
    bus_id: int = -1
    bus_occupancy_at_board: float = 0.0
    scenario: str = ""
    boarded: bool = False


@dataclass
class ReplicationResult:
    """Summary statistics from a single simulation replication."""
    replication: int
    scenario: str
    avg_wait_time: float
    median_wait_time: float
    max_occupancy: float
    avg_load_factor: float
    load_factor_cv: float
    total_boarded: int
    total_refused: int
    total_arrived: int


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_peak_time(t: float) -> bool:
    """Return True if simulation time t is in a peak period."""
    for start, end in cfg.PEAK_HOURS:
        if start <= t <= end:
            return True
    return False


def mnl_boarding_prob(
    wait_time_minutes: float,
    occupancy_ratio: float,
    is_peak: bool,
    noise: float = 0.0,
    rng: np.random.Generator = None,
) -> float:
    """
    Compute MNL probability of boarding the current bus.

    Utility of waiting = β_wait·t + β_crowding + β_peak·I_peak
    P(board) = 1 - P(wait) = sigmoid(-U_wait)

    Parameters
    ----------
    wait_time_minutes : float
        Expected wait for next bus (minutes).
    occupancy_ratio : float
        Current bus occupancy as fraction [0, 1].
    is_peak : bool
        True if current time is a peak period.
    noise : float
        Std dev of uniform noise on occupancy perception (0 = perfect info).
    rng : np.random.Generator, optional
        RNG for noise injection.

    Returns
    -------
    float
        Probability in [0, 1] of choosing to board now.
    """
    # Apply perceptual noise to occupancy
    perceived = occupancy_ratio
    if noise > 0 and rng is not None:
        perceived = np.clip(occupancy_ratio + rng.uniform(-noise, noise), 0.0, 1.0)

    is_packed = perceived > 0.80
    is_standing = 0.50 < perceived <= 0.80

    u_wait = (
        cfg.BETA_WAIT * wait_time_minutes
        + (cfg.BETA_CROWDING_PACKED if is_packed else 0.0)
        + (cfg.BETA_CROWDING_STANDING if is_standing else 0.0)
        + (cfg.BETA_PEAK if is_peak else 0.0)
    )

    p_wait = 1.0 / (1.0 + math.exp(-u_wait))
    p_board = 1.0 - p_wait
    return float(np.clip(p_board, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SimPy processes
# ---------------------------------------------------------------------------

def bus_route_process(
    env: simpy.Environment,
    bus_id: int,
    scenario: str,
    waiting_passengers: List[List["PassengerRecord"]],
    load_factors: List[float],
    rng: np.random.Generator,
) -> Generator:
    """
    SimPy process representing a single bus travelling the route.

    The bus visits each stop, loads waiting passengers (subject to
    boarding decision under the given scenario), drops off alighting
    passengers, and records load factor.

    Parameters
    ----------
    env : simpy.Environment
    bus_id : int
    scenario : str
        One of 'no_info', 'perfect_info', 'imperfect_info'.
    waiting_passengers : list of lists
        waiting_passengers[stop_id] = list of PassengerRecord waiting.
    load_factors : list
        Shared list to append observed load factors.
    rng : np.random.Generator
    """
    occupancy = 0
    capacity = cfg.BUS_CAPACITY

    for stop_id in range(cfg.NUM_STOPS):
        # Dwell at stop
        yield env.timeout(cfg.BUS_DWELL_TIME)

        current_time = env.now
        peak = is_peak_time(current_time)

        # --- Alighting ---
        alight_count = 0
        if occupancy > 0 and stop_id < cfg.NUM_STOPS - 1:
            # Each passenger alights with given probability
            alight_count = min(
                occupancy,
                rng.binomial(occupancy, cfg.ALIGHT_PROBABILITY)
            )
            occupancy -= alight_count

        # --- Boarding decision ---
        boarders = []
        refused = []
        for pax in waiting_passengers[stop_id]:
            if pax.board_time > 0:
                continue  # already boarded an earlier bus

            if occupancy >= capacity:
                refused.append(pax)
                continue

            should_board = True
            if scenario != "no_info":
                # Estimate wait for next bus
                expected_next_wait = rng.exponential(cfg.MEAN_HEADWAY)
                occ_ratio = occupancy / capacity
                noise = cfg.NOISE_LEVEL if scenario == "imperfect_info" else 0.0
                p_board = mnl_boarding_prob(
                    expected_next_wait, occ_ratio, peak, noise, rng
                )
                should_board = rng.random() < p_board

            if should_board and occupancy < capacity:
                occupancy += 1
                pax.board_time = current_time
                pax.wait_time = current_time - pax.arrival_time
                pax.bus_id = bus_id
                pax.bus_occupancy_at_board = occupancy / capacity
                pax.boarded = True
                boarders.append(pax)

        # Record load factor at this stop
        load_factors.append(occupancy / capacity)

    # Yield control back
    yield env.timeout(0)


def passenger_arrival_process(
    env: simpy.Environment,
    stop_id: int,
    waiting: List["PassengerRecord"],
    passenger_counter: List[int],
    rng: np.random.Generator,
) -> Generator:
    """
    SimPy process: generates passengers at a bus stop via Poisson arrivals.

    Parameters
    ----------
    env : simpy.Environment
    stop_id : int
    waiting : list
        Shared waiting list for this stop.
    passenger_counter : list
        Single-element list used as mutable integer counter.
    rng : np.random.Generator
    """
    while True:
        # Inter-arrival time ~ Exponential(1/lambda)
        inter_arrival = rng.exponential(1.0 / cfg.PASSENGER_ARRIVAL_RATE)
        yield env.timeout(inter_arrival)

        pax = PassengerRecord(
            passenger_id=passenger_counter[0],
            stop_id=stop_id,
            arrival_time=env.now,
        )
        passenger_counter[0] += 1
        waiting.append(pax)


def bus_dispatch_process(
    env: simpy.Environment,
    scenario: str,
    waiting_passengers: List[List["PassengerRecord"]],
    load_factors: List[float],
    rng: np.random.Generator,
) -> Generator:
    """
    SimPy process: dispatches buses at Poisson-distributed headways.

    Parameters
    ----------
    env : simpy.Environment
    scenario : str
    waiting_passengers : list of lists
    load_factors : list
    rng : np.random.Generator
    """
    bus_id = 0
    while True:
        headway = rng.exponential(cfg.MEAN_HEADWAY)
        yield env.timeout(headway)
        env.process(
            bus_route_process(
                env, bus_id, scenario, waiting_passengers, load_factors, rng
            )
        )
        bus_id += 1


# ---------------------------------------------------------------------------
# Single replication runner
# ---------------------------------------------------------------------------

def run_replication(
    scenario: str,
    replication: int,
    rng: np.random.Generator,
) -> ReplicationResult:
    """
    Run one simulation replication for the given scenario.

    Parameters
    ----------
    scenario : str
        'no_info', 'perfect_info', or 'imperfect_info'.
    replication : int
        Replication index (for logging).
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    ReplicationResult
    """
    env = simpy.Environment()

    # Per-stop waiting lists
    waiting_passengers: List[List[PassengerRecord]] = [[] for _ in range(cfg.NUM_STOPS)]
    load_factors: List[float] = []
    passenger_counter = [0]

    # Start passenger arrival processes at each stop
    for stop_id in range(cfg.NUM_STOPS):
        env.process(
            passenger_arrival_process(
                env, stop_id, waiting_passengers[stop_id], passenger_counter, rng
            )
        )

    # Start bus dispatch process
    env.process(
        bus_dispatch_process(env, scenario, waiting_passengers, load_factors, rng)
    )

    # Run simulation
    env.run(until=cfg.SIM_DURATION)

    # Collect all passengers (after warm-up)
    all_pax = [
        pax
        for stop_list in waiting_passengers
        for pax in stop_list
        if pax.arrival_time >= cfg.WARM_UP_PERIOD
    ]

    boarded = [p for p in all_pax if p.boarded]
    refused = [p for p in all_pax if not p.boarded]

    wait_times = [p.wait_time for p in boarded] if boarded else [0.0]
    lf_after_warmup = load_factors[len(load_factors) // 4:]  # discard first quarter

    avg_lf = float(np.mean(lf_after_warmup)) if lf_after_warmup else 0.0
    lf_cv = float(np.std(lf_after_warmup) / avg_lf) if avg_lf > 0 else 0.0

    return ReplicationResult(
        replication=replication,
        scenario=scenario,
        avg_wait_time=float(np.mean(wait_times)),
        median_wait_time=float(np.median(wait_times)),
        max_occupancy=float(max(lf_after_warmup, default=0.0)),
        avg_load_factor=avg_lf,
        load_factor_cv=lf_cv,
        total_boarded=len(boarded),
        total_refused=len(refused),
        total_arrived=len(all_pax),
    )


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

def run_all_scenarios(
    scenarios: List[str] = None,
    n_reps: int = None,
    output_dir: str = None,
    seed: int = None,
) -> "pd.DataFrame":
    """
    Run all scenarios × replications and save results to CSV.

    Parameters
    ----------
    scenarios : list of str, optional
        Scenarios to run. Defaults to all three.
    n_reps : int, optional
        Number of replications. Defaults to cfg.NUM_REPLICATIONS.
    output_dir : str, optional
        Output directory. Defaults to cfg.OUTPUT_DIR.
    seed : int, optional
        Base random seed. Defaults to cfg.RANDOM_SEED.

    Returns
    -------
    pd.DataFrame
        Results DataFrame with one row per (scenario, replication).
    """
    if scenarios is None:
        scenarios = SCENARIOS
    if n_reps is None:
        n_reps = cfg.NUM_REPLICATIONS
    if output_dir is None:
        output_dir = cfg.OUTPUT_DIR
    if seed is None:
        seed = cfg.RANDOM_SEED if cfg.RANDOM_SEED is not None else 0

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(scenarios) * n_reps

    log.info(f"Running {total} replications ({len(scenarios)} scenarios × {n_reps} reps)...")

    for s_idx, scenario in enumerate(scenarios):
        log.info(f"Scenario {s_idx+1}/{len(scenarios)}: {scenario}")
        for rep in range(n_reps):
            rng = np.random.default_rng(seed + s_idx * 10000 + rep)
            result = run_replication(scenario, rep, rng)
            results.append(result)
            if (rep + 1) % 25 == 0:
                log.info(f"  Rep {rep+1}/{n_reps} done")

    df = pd.DataFrame([vars(r) for r in results])

    csv_path = out_dir / cfg.RESULTS_FILENAME
    df.to_csv(csv_path, index=False)
    log.info(f"Results saved: {csv_path}")

    # Print summary
    summary = df.groupby("scenario")[["avg_wait_time", "load_factor_cv", "total_boarded"]].mean()
    log.info(f"\nSummary:\n{summary.to_string()}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sanash Bus Occupancy SimPy Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--replications", "-r",
        type=int,
        default=cfg.NUM_REPLICATIONS,
        help=f"Replications per scenario (default: {cfg.NUM_REPLICATIONS})",
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=SCENARIOS + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default=cfg.OUTPUT_DIR,
        help=f"Output directory (default: {cfg.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.RANDOM_SEED,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run simulation."""
    args = parse_args()
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    run_all_scenarios(
        scenarios=scenarios,
        n_reps=args.replications,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
