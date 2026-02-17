"""
Discrete-event simulation of a simple bus route using SimPy.

Components:
- Environment (simpy.Environment)
- Passenger generator: Poisson process of arrivals at a bus stop.
- Bus with limited capacity, cycling through the stop on a fixed headway.

This is a minimal but extensible base for more realistic transit
agent-based modelling in the Sanash project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import simpy


@dataclass
class Passenger:
    """Passenger waiting at a bus stop."""

    id: int
    arrival_time: float


class BusStop:
    """A single bus stop maintaining a FIFO queue of passengers."""

    def __init__(self, env: simpy.Environment, name: str = "Stop-1") -> None:
        self.env = env
        self.name = name
        self.queue: List[Passenger] = []
        self.total_served: int = 0
        self.total_lost: int = 0  # e.g., due to full buses (optional)

    def add_passenger(self, passenger: Passenger) -> None:
        self.queue.append(passenger)
        print(
            f"[{self.env.now:6.2f}] Passenger {passenger.id} arrived at {self.name}; "
            f"queue length={len(self.queue)}"
        )

    def board_passengers(self, capacity: int) -> List[Passenger]:
        """Board up to `capacity` passengers, returning the list that boarded."""
        num_to_board = min(capacity, len(self.queue))
        boarded = self.queue[:num_to_board]
        self.queue = self.queue[num_to_board:]

        for p in boarded:
            self.total_served += 1
            wait_time = self.env.now - p.arrival_time
            print(
                f"[{self.env.now:6.2f}] Passenger {p.id} boards; "
                f"wait_time={wait_time:.2f} min; remaining_queue={len(self.queue)}"
            )

        if capacity < len(self.queue):
            # Not used for anything yet, but a hook for metrics about lost demand.
            self.total_lost += len(self.queue) - capacity

        return boarded


def passenger_arrival_process(
    env: simpy.Environment,
    stop: BusStop,
    arrival_rate: float,
) -> None:
    """
    Generate passengers as a Poisson process.

    arrival_rate: expected number of arrivals per minute.
    Inter-arrival times are exponential with mean 1 / arrival_rate.
    """
    passenger_id = 0

    while True:
        # Exponential inter-arrival times (Poisson process).
        if arrival_rate <= 0:
            # No arrivals if rate is non-positive
            yield env.timeout(1.0)
            continue

        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        yield env.timeout(inter_arrival)

        passenger_id += 1
        p = Passenger(id=passenger_id, arrival_time=env.now)
        stop.add_passenger(p)


def bus_process(
    env: simpy.Environment,
    stop: BusStop,
    name: str,
    capacity: int,
    headway: float,
    dwell_time: float,
) -> None:
    """
    Bus cycling between runs on a single stop with fixed headway.

    Parameters
    ----------
    headway : float
        Time between departures in minutes.
    dwell_time : float
        Time spent at the stop for boarding/alighting.
    """
    run = 0
    while True:
        run += 1
        print(
            f"[{env.now:6.2f}] {name} arrives for run {run}; "
            f"queue length={len(stop.queue)}"
        )

        # Boarding
        boarded = stop.board_passengers(capacity)
        print(
            f"[{env.now:6.2f}] {name} boarded {len(boarded)} passengers "
            f"(capacity={capacity})"
        )

        # Dwell time at stop
        yield env.timeout(dwell_time)

        # Travel until next arrival (headway minus dwell at stop)
        cruise_time = max(0.0, headway - dwell_time)
        print(f"[{env.now:6.2f}] {name} departs; cruising for {cruise_time:.2f} min")
        yield env.timeout(cruise_time)


def run_simulation(
    sim_time: float = 120.0,
    arrival_rate: float = 0.8,
    bus_capacity: int = 40,
    headway: float = 10.0,
    dwell_time: float = 1.0,
) -> None:
    """
    Run a full simulation of a single bus stop and route.

    Parameters
    ----------
    sim_time : float
        Total simulation time in minutes.
    arrival_rate : float
        Passenger arrival rate (Poisson) in passengers per minute.
    bus_capacity : int
        Maximum number of passengers the bus can carry.
    headway : float
        Time between bus departures in minutes.
    dwell_time : float
        Time the bus spends at the stop for boarding.
    """
    env = simpy.Environment()
    stop = BusStop(env, name="Central-Stop")

    # Start processes
    env.process(passenger_arrival_process(env, stop, arrival_rate=arrival_rate))
    env.process(
        bus_process(
            env,
            stop=stop,
            name="Bus-1",
            capacity=bus_capacity,
            headway=headway,
            dwell_time=dwell_time,
        )
    )

    print(
        f"Starting simulation for {sim_time} minutes. "
        f"λ={arrival_rate} pax/min, capacity={bus_capacity}, headway={headway} min."
    )
    env.run(until=sim_time)

    print("\nSimulation finished.")
    print(f"Total passengers served: {stop.total_served}")
    print(f"Passengers remaining in queue: {len(stop.queue)}")
    if sim_time > 0:
        theoretical_arrivals = arrival_rate * sim_time
        print(
            f"Approx. theoretical arrivals (Poisson mean): {theoretical_arrivals:.1f}"
        )


if __name__ == "__main__":
    # Basic demo run for quick inspection.
    run_simulation(
        sim_time=120.0,
        arrival_rate=0.8,
        bus_capacity=40,
        headway=10.0,
        dwell_time=1.0,
    )

