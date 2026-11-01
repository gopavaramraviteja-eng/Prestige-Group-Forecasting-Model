# Synthetic Construction Progress & Cost Dataset

**Purpose:** Fully synthetic data for prototyping a construction progress / cost forecasting model for a residential real-estate developer. No real project, financial, or personnel data is used anywhere in this dataset — project names, budgets, timelines, and events are all randomly generated in a style resembling a South-India-focused residential developer (the kind of portfolio referenced on the prestigeconstructions.com ad page), purely so the schema and value ranges feel realistic for modeling purposes.

Generated: 2026-09-06. Daily granularity. Window: 2024-09-06 to 2026-09-06 (2 years).
Random seed: 42 (re-running `gen_data.py` reproduces the same data).

## Files

### 1. `dim_projects.csv` (one row per project, 18 projects)
| Column | Description |
|---|---|
| project_id | Unique project key (PRJ001…PRJ018) |
| project_name | Synthetic project name |
| city | City (Bengaluru, Chennai, Hyderabad, Pune, Mumbai, Kochi) |
| locality | Locality/micro-market within the city |
| project_type | High-rise Apartments / Premium Apartments / Villas / Plotted Development |
| total_units | Number of saleable units |
| avg_unit_size_sqft | Average unit size |
| saleable_area_sqft | total_units × avg_unit_size_sqft |
| total_towers | Number of towers/blocks |
| planned_start_date | Originally scheduled construction start |
| actual_start_date | Actual mobilization date (may be delayed vs. planned) |
| planned_duration_months | Planned construction duration |
| planned_completion_date | planned_start_date + planned duration |
| approved_budget_cr | Originally approved construction budget (INR crore) |
| cost_per_sqft_inr | Construction cost per sqft used to derive the budget |
| overrun_factor | Multiplier applied to approved budget to get the revised/actual budget (captures cost overrun) |
| mgmt_quality_factor | Latent per-project execution-efficiency factor (0.82–1.06) driving how close actual pace tracks planned pace |
| status_as_of_today | Ongoing / Completed as of the last date in the window |
| revised_budget_cr | approved_budget_cr × overrun_factor — the effective budget actually being spent against |

### 2. `fact_daily_construction_progress.csv` (~9,850 rows — one row per project per active construction day)
This is the main table for a progress/cost forecasting model.

| Column | Description |
|---|---|
| date | Calendar date |
| project_id | Foreign key to dim_projects |
| city | Denormalized for convenience |
| planned_progress_pct | Cumulative % complete per the original schedule (S-curve) |
| actual_progress_pct | Cumulative % physically complete (ground truth to forecast) |
| daily_progress_increment_pct | That day's incremental physical progress |
| milestone_stage | Current construction phase: Excavation & Foundation / Basement & Podium / Superstructure / MEP & Finishing / Handover & Clearance |
| daily_cost_incurred_lakhs | Cost incurred that day (INR lakh) |
| cumulative_cost_incurred_cr | Cumulative cost incurred to date (INR crore) |
| planned_cost_to_date_cr | Cumulative cost per the original budget/schedule (INR crore) |
| revised_budget_cr | Total revised budget for the project (constant per project) |
| workers_onsite | Estimated daily workforce headcount |
| rainfall_mm | Simulated daily rainfall (drives monsoon slowdowns, city- and season-specific) |
| material_price_index | Macro construction material price index (base 100, inflates ~7%/yr with cyclical + random noise), common across projects on a given date |
| is_delay_event | 1 if a disruption occurred that day |
| delay_reason | Heavy Rainfall / Labor Shortage / Material Delay / Regulatory-Approval Hold / Equipment Breakdown / Fund Flow Delay / blank |

### 3. `fact_daily_cost_breakdown.csv` (~49,000 rows — one row per project per active day per cost category)
Splits `daily_cost_incurred_lakhs` from the progress fact table into categories: **Land & Approvals**, **Materials**, **Labor**, **Equipment & Machinery**, **Overheads & Marketing**. Summing categories per (date, project_id) reconciles exactly to `daily_cost_incurred_lakhs` in the progress table.

### 4. `dim_milestones.csv` (72 rows — 4 milestones × 18 projects)
| Column | Description |
|---|---|
| project_id | Foreign key |
| milestone_name | Foundation Complete / Superstructure Complete / Finishing Complete / Handover |
| planned_date | Scheduled date per original plan |
| actual_date | Date actually achieved (blank if not yet reached as of window end) |
| variance_days | actual_date − planned_date, in days (positive = delayed) |

## How the data was generated (key modeling choices)

- **Planned schedule:** each project's *planned* cumulative progress follows a logistic S-curve (slow mobilization → fast structural phase → slow finishing) scaled to its planned duration.
- **Actual progress:** driven independently by a phase-dependent base daily rate (so a delayed project keeps a realistic, if slow, capacity to eventually finish, rather than stalling forever once past its nominal schedule), multiplied by:
  - a per-project `mgmt_quality_factor` (persistent execution efficiency),
  - a rainfall/monsoon factor (city- and season-specific, sharply reducing progress on very wet days),
  - day-to-day random noise,
  - occasional stoppage events (labor shortage, material delay, regulatory hold, equipment breakdown, fund-flow delay), ~4.5% of days.
- **Cost:** daily cost incurred is proportional to that day's physical progress, scaled by a phase cost multiplier (foundation and handover are cheaper per % than superstructure/finishing) and the macro material price index. A per-project `overrun_factor` (1.00–1.28×) inflates the effective budget being spent against, producing realistic cost overruns.
- **Result:** across the 18 projects you'll see a full spread of construction phases and outcomes as of 2026-09-06 — 4 fully completed, several 60–90% complete, several early-stage — plus realistic monsoon seasonality, cost overruns, and schedule slippage (milestones average ~5-6 months behind plan), which should give a forecasting model real signal to learn from.

## Suggested forecasting use cases
- Predict `actual_progress_pct` (or daily increment) N days ahead per project, using rainfall, workers_onsite, material_price_index, and delay flags as features.
- Predict `cumulative_cost_incurred_cr` / final cost-at-completion vs. `revised_budget_cr` (cost-overrun forecasting).
- Predict milestone slippage (`variance_days`) from early-stage indicators.
- Portfolio-level: aggregate daily cost/progress across projects per city to forecast citywide construction spend.
