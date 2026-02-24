# Sanash Research Reproduction Guide

This guide explains how to reproduce each research phase of the Sanash project, from computer vision training to survey analysis to agent-based simulation.

---

## Phase 1: Computer Vision — CSRNet Crowd Counting

### Dataset
- **ShanghaiTech Part B** — 716 train / 316 test images
- Location: `data/shanghaitech/part_B_final/`
- Ground truth: `.mat` files with point annotations in `ground_truth/` subdirectories

### Preprocessing
```bash
# Convert .mat point annotations to H5 density maps:
python data_pipeline/preprocess_shanghaitech.py --part B --split train
python data_pipeline/preprocess_shanghaitech.py --part B --split test
```

### Training
**Google Colab (recommended):**
1. Open `colab_training.ipynb` in Colab
2. Mount Google Drive, set `DATA_PATH` to your dataset location
3. Run all cells — trains CSRNet with Adam optimizer, ~100 epochs

**Local training:**
```bash
pip install torch torchvision
python trainers/training/train_csrnet.py --data-dir data/shanghaitech/part_B_final/ --epochs 100
```

### Evaluation Metrics
- **MAE** (Mean Absolute Error) — primary metric
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)

Target performance on Part B: MAE < 15, RMSE < 25 (state-of-art is ~7.2 MAE)

### Reproducing Paper Figures

**Figure: Density Map Comparison (3-panel)**
```bash
python scripts/visualize_predictions.py \
  --model-path models/csrnet/ \
  --data-dir data/shanghaitech/part_B_final/test_data/ \
  --num-images 10 \
  --output-dir output/figures/density_maps/
```
Output: `output/figures/density_maps/density_map_IMG_*.png`

**Figure: Predicted vs GT Scatter Plot**
```bash
python scripts/generate_scatter_plot.py \
  --input output/eval_results.csv \
  --output output/figures/scatter_pred_vs_gt.png
```
*If `eval_results.csv` doesn't exist, synthetic data is used for demonstration.*

---

## Phase 2: Discrete Choice Experiment (Survey)

### Methodology
- **Type:** Binary DCE (Discrete Choice Experiment)
- **Population:** Almaty bus commuters, 18+, using buses ≥1×/week
- **Sample:** n = 167 respondents
- **Collection:** Google Forms, March 2025

### Survey Instrument
- `survey/instrument/survey_questions_en.md` — English
- `survey/instrument/survey_questions_ru.md` — Russian
- `survey/instrument/survey_questions_kz.md` — Kazakh

### DCE Design
6 scenarios, balanced orthogonal design varying:
| Attribute | Levels |
|-----------|--------|
| Crowding | Seats (<50%), Standing (50–80%), Packed (>80%) |
| Wait time for next bus | 2, 5, 10, 15 minutes |
| Time of day | Peak / Off-peak |

### Statistical Model
Binary logit (Willingness-to-Wait):

```
U(wait) = β_wait·t + β_packed·I_packed + β_standing·I_standing + β_peak·I_peak + ε
P(wait) = sigmoid(U_wait)
WTW = |β_crowding / β_wait|  (minutes)
```

### Running the Analysis
```bash
pip install -r survey/analysis/requirements.txt

# With real response data:
python survey/analysis/survey_analysis.py --input survey/data/responses.csv

# Demo mode (synthetic data, no CSV needed):
python survey/analysis/survey_analysis.py
```

**Output** (saved to `survey/output/`):
- `demographics.png` — Respondent demographics
- `wtw_chart.png` — Willingness-to-Wait bar chart (Figure 5 in paper)
- `choice_proportions.png` — Board vs wait by scenario
- `model_coefficients.csv` — MNL coefficient table
- `wtw_results.csv` — WTW estimates

### Expected Results
- WTW (packed bus): ~8 minutes
- WTW (standing room): ~4 minutes
- β_wait significantly negative (p < 0.001)

---

## Phase 3: Agent-Based Simulation (SimPy)

### Model Description
- **Route:** 8 bus stops, single direction
- **Buses:** 60-seat capacity, Poisson headways (μ=10 min)
- **Passengers:** Poisson arrivals (3 pax/min/stop)
- **Decision:** MNL model with DCE-calibrated coefficients

### Three Information Scenarios
| Scenario | Behaviour |
|----------|-----------|
| `no_info` | Board any bus with available space |
| `perfect_info` | Observe exact occupancy, decide via MNL |
| `imperfect_info` | Observe occupancy with ±10% noise, decide via MNL |

### Running the Simulation
```bash
pip install -r simulation/requirements.txt

# Full run: 100 reps × 3 scenarios (≈ 5 min):
python simulation/bus_simulation.py

# Quick test (10 reps):
python simulation/bus_simulation.py --replications 10

# Single scenario:
python simulation/bus_simulation.py --scenario perfect_info --replications 50
```

Output: `simulation/output/simulation_results.csv`

### Post-Simulation Analysis
```bash
python simulation/analysis.py
```

**Output** (saved to `simulation/output/`):
- `wait_times_boxplot.png` — Wait time by scenario (Figure 6)
- `load_factor_cv.png` — Load distribution equity
- `occupancy_timeseries.png` — Peak occupancy over replications
- `sensitivity_tornado.png` — Sensitivity to coefficient changes
- `anova_results.txt` — ANOVA + Bonferroni table

### Expected Results
| Metric | no_info | perfect_info | imperfect_info |
|--------|---------|--------------|----------------|
| Avg wait (min) | ~6.5 | ~4.1 | ~4.8 |
| Load factor CV | ~0.35 | ~0.18 | ~0.23 |
| ANOVA p-value | < 0.001 (all pairwise significant) | | |

---

## Running Everything End-to-End

```bash
# 1. Install all dependencies
pip install pandas statsmodels matplotlib seaborn scipy simpy numpy

# 2. Survey analysis
python survey/analysis/survey_analysis.py

# 3. Simulation
python simulation/bus_simulation.py --replications 10
python simulation/analysis.py

# 4. CV visualizations
python scripts/visualize_predictions.py
python scripts/generate_scatter_plot.py

# 5. Run tests
pip install pytest
pytest tests/ -v
```

---

## File Locations Quick Reference

| Asset | Location |
|-------|----------|
| Survey instrument (EN) | `survey/instrument/survey_questions_en.md` |
| Survey analysis | `survey/analysis/survey_analysis.py` |
| Survey output | `survey/output/` |
| SimPy simulation | `simulation/bus_simulation.py` |
| Simulation config | `simulation/config.py` |
| Simulation output | `simulation/output/` |
| Density map viz | `scripts/visualize_predictions.py` |
| Scatter plot | `scripts/generate_scatter_plot.py` |
| Figure output | `output/figures/` |
| ShanghaiTech dataset | `data/shanghaitech/` |
| CSRNet model | `models/csrnet/` |
