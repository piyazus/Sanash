"""Sanash Bus Simulation — Configuration Parameters."""

# Route configuration
NUM_STOPS = 8
BUS_CAPACITY = 60         # passengers
MEAN_HEADWAY = 10         # minutes between buses (Poisson headway)
PASSENGER_ARRIVAL_RATE = 3  # passengers per minute per stop

# MNL utility coefficients (calibrated to DCE survey results)
BETA_WAIT = -0.15             # disutility per minute of wait
BETA_CROWDING_PACKED = -1.2   # disutility from packed bus (>80% full)
BETA_CROWDING_STANDING = -0.6 # disutility from standing room (50-80% full)
BETA_PEAK = -0.3              # additional disutility during peak hours

# Imperfect information noise
NOISE_LEVEL = 0.10  # ± uniform noise on perceived occupancy ratio

# Simulation settings
NUM_REPLICATIONS = 100
SIM_DURATION = 240   # minutes (4 hours)
WARM_UP_PERIOD = 30  # minutes to discard from stats
RANDOM_SEED = 42

# Bus operations
BUS_DWELL_TIME = 0.5    # minutes stopped at each stop
ALIGHT_PROBABILITY = 0.2  # probability each passenger alights per stop

# Peak hour windows [start, end] in simulation minutes
PEAK_HOURS = [(0, 60), (180, 240)]  # first/last hour = peak

# Output
OUTPUT_DIR = "simulation/output"
RESULTS_FILENAME = "simulation_results.csv"
