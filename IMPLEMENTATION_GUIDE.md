# Implementation Walkthrough
## Smart Energy Consumption Forecasting — COM748 Masters Research
### Ali Haider

This document explains what is happening in every step of the implementation,
what each file does, and what every result means — written in plain English.

---

## What This Research Is About

- We want to predict future electricity consumption using historical data
- We compare 5 different prediction models to find which one works best
- We use real electricity data from the PJM East regional grid in the USA
- The grid covers millions of homes and businesses across the eastern United States

---

## The Data

This research uses **two real-world electricity datasets**. PJM is the primary dataset used for all main results and conclusions. UCI is used as a secondary comparison to test whether the models behave differently on household-level data.

---

### Dataset 1 — PJM East Regional Grid (Primary)
**File:** `data/raw/PJME_hourly.csv`

| Property | Value |
|----------|-------|
| Source | PJM Interconnection LLC (downloaded from Kaggle) |
| What it covers | The PJM East region — a large electricity grid serving tens of millions of people across the eastern United States |
| Time period | 2002-12-31 to 2018-01-02 (approximately 16 years) |
| Frequency | Hourly — one reading every hour |
| Number of rows | 145,366 rows |
| Number of columns | 2 |

**Columns in this file:**

| Column name | Type | Example value | What it means |
|-------------|------|--------------|---------------|
| `Datetime` | timestamp | `2002-12-31 01:00:00` | The date and hour this reading was recorded |
| `PJME_MW` | number | `26498.0` | Total electricity demand in megawatts (MW) for that hour |

**What the values look like:**
- Minimum value: around 19,000 MW (cold, quiet winter nights)
- Maximum value: around 65,000 MW (hot summer afternoons with air conditioning)
- Typical daytime value: 40,000–55,000 MW on a weekday
- 1 megawatt = enough electricity for roughly 400–1,000 average homes

**Why this dataset was chosen:**
- Very long record (16 years) with reliable hourly readings
- Clear and predictable daily/seasonal patterns — ideal for testing forecasting models
- Widely used in academic energy forecasting research for comparison

---

### Dataset 2 — UCI Household Power Consumption (Secondary)
**File:** `data/raw/household_power_consumption.txt` (also saved as `.csv`)

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| What it covers | A single house in Sceaux, France — minute-by-minute power readings |
| Time period | December 2006 to November 2010 (approximately 4 years) |
| Frequency | Every minute (resampled to hourly in preprocessing) |
| Number of rows | ~2 million rows (minute-level); ~34,000 after resampling to hourly |
| Number of columns | 9 |

**Columns in this file:**

| Column name | Unit | Example value | What it means |
|-------------|------|--------------|---------------|
| `Date` | date | `16/12/2006` | Date of the reading |
| `Time` | time | `17:24:00` | Time of the reading |
| `Global_active_power` | kilowatts (kW) | `4.216` | **Total electricity the house is using** — this is the main forecast target |
| `Global_reactive_power` | kVAr | `0.418` | Reactive power (related to motors and inductors — not directly consumed) |
| `Voltage` | volts | `234.84` | Voltage of the supply (should be around 230–240V) |
| `Global_intensity` | amperes | `18.4` | Current flowing into the house |
| `Sub_metering_1` | watt-hours | `0.0` | Kitchen: dishwasher, oven, microwave |
| `Sub_metering_2` | watt-hours | `1.0` | Laundry: washing machine, dryer, refrigerator, light |
| `Sub_metering_3` | watt-hours | `17.0` | Water heater and air conditioner |

**What the values look like:**
- Global_active_power ranges from 0.076 kW to 11.1 kW
- 0 = nothing switched on; 10+ = many appliances running simultaneously
- Very erratic: a single person turning on a kettle causes a spike that no model can predict

**Why this dataset is harder to forecast:**
- Individual human behaviour is unpredictable — a family goes on holiday, consumption drops to near zero for two weeks
- No pattern at the regional grid level — the UCI dataset reflects one person's habits
- Result: all 5 models perform significantly worse on UCI than on PJM — this is an expected and meaningful finding, not a failure

---

### Processed Data Files
After Notebook 02 (preprocessing) runs, cleaned versions of both datasets are saved:

**`data/processed/pjm_processed.csv`** — The cleaned PJM data with extra columns added:

| Column | Example | What it means |
|--------|---------|---------------|
| `Datetime` | `2002-01-01 01:00:00` | Timestamp |
| `PJME_MW` | `30393.0` | Original MW value |
| `hour` | `1` | Hour of day (0–23) |
| `day_of_week` | `1` | Day of week (0=Monday, 6=Sunday) |
| `day_of_month` | `1` | Day of month (1–31) |
| `month` | `1` | Month (1–12) |
| `year` | `2002` | Year |
| `is_weekend` | `0` | 1 if Saturday/Sunday, 0 if weekday |
| `season` | `0` | 0=Winter, 1=Spring, 2=Summer, 3=Autumn |
| `PJME_MW_scaled` | `0.4624` | MW value scaled to 0–1 range (this is what models see) |

**`data/processed/uci_processed.csv`** — The cleaned UCI data with the same time features plus all 8 power columns:
- Key column used for forecasting: `GAP_scaled` (Global_active_power scaled to 0–1)

**`data/processed/pjm_sequences.npz`** and **`uci_sequences.npz`** — NumPy binary files (cannot open in Excel):
- `X_train`: training inputs — shape `(116,277, 24)` for PJM — meaning 116,277 training examples, each being 24 consecutive hours
- `y_train`: training targets — the next hour's value for each training example
- `X_test` / `y_test`: same structure but for the test period (2015–2018)

**`data/processed/pjm_scaler.pkl`** and **`uci_scaler.pkl`** — Python objects that store the Min-Max scaling settings:
- These are used to convert predictions back from the 0–1 range to real MW or kW values
- Without this file, the predictions cannot be converted back to real units

---

## Step 1 — Data Preprocessing
### Notebook: `notebooks/02_preprocessing.ipynb`

**What is preprocessing?**
Raw data is messy. Before training any model, we must clean and prepare it.

**What this notebook does, step by step:**

- **Loads the CSV file** — reads the 145,366 rows into memory
- **Removes duplicate timestamps** — some hours appear twice in the data due to recording errors; we keep only the first occurrence
- **Fills missing hours** — some hours are missing entirely (e.g. the data jumps from 2am to 4am); we fill the gap by copying the nearest available value
- **Removes outliers** — some readings are clearly wrong (e.g. 0 MW or extremely high spikes); we clip these using the IQR method (anything too far above or below the average is trimmed)
- **Adds time features** — we add extra columns that help the model understand patterns:
  - `hour` — what hour of the day (0 to 23)
  - `day_of_week` — Monday to Sunday (0 to 6)
  - `month` — January to December (1 to 12)
  - `season` — Winter, Spring, Summer, Autumn
  - `is_weekend` — 1 if Saturday or Sunday, 0 if weekday
- **Normalises the data** — scales all values between 0 and 1 so the models train faster and more fairly. The original MW values can be recovered afterwards.
- **Creates sequences** — splits the data into input-output pairs:
  - Input (X): the last 24 hours of electricity usage
  - Output (y): the next hour's usage
  - This is how the model learns: "given what happened in the last 24 hours, predict the next hour"
- **Splits into training and testing:**
  - First 80% of data (2002–2015) → used to train the models
  - Last 20% of data (2015–2018) → used to test how well they predict

**Files saved after this step:**
- `data/processed/pjm_processed.csv` — cleaned data with all time features added
- `data/processed/pjm_sequences.npz` — the input/output sequences ready for model training
- `data/processed/pjm_scaler.pkl` — saves the normalisation settings so we can reverse it later

---

## Step 2 — Exploratory Data Analysis (EDA)
### Notebook: `notebooks/03_eda.ipynb`

**What is EDA?**
Before training any model, we study the data visually and statistically to understand its patterns and behaviour. EDA answers the question: "what is actually in this data, and what structure can a model learn from?" Every chart produced here either justifies a design decision or provides findings that go directly into the research report.

This notebook runs EDA on **both datasets** — PJM East and UCI Household — so we can compare a regional grid against a single household.

---

### EDA Analysis 1 — Time Series Plot
**Chart saved:** `notebooks/figures/pjm_timeseries.png` and `uci_timeseries.png`

**What it is:** A simple line graph plotting every hourly reading across the full dataset from start to finish.

**What we do:** Plot `PJME_MW` (y-axis) against `Datetime` (x-axis) for all 145,366 PJM rows.

**What we see in PJM:**
- Demand ranges from roughly 19,000 MW at the quietest winter nights to 65,000 MW at peak summer afternoons
- Clear annual seasonality: two humps per year — one summer peak (air conditioning) and one winter peak (heating)
- A slight upward trend from 2002 to ~2010, then levelling off — reflecting regional economic growth and energy efficiency improvements
- Short-term spikes visible even at the zoomed-out scale — these are weather-driven events (heat waves, cold snaps)

**What we see in UCI:**
- Demand is far more erratic — 0.076 kW (essentially off) to 11.1 kW (many appliances on)
- No clear upward trend over 4 years
- Much harder to see any regular pattern because individual human behaviour creates noise

**What this tells us for the research:**
- PJM has clear, learnable structure — it is a good candidate for forecasting
- UCI is unpredictable at the hourly level — any model trained on it will have higher error, which is an expected and publishable finding

---

### EDA Analysis 2 — Seasonal Decomposition
**Chart saved:** `notebooks/figures/pjm_decomposition.png` and `uci_decomposition.png`

**What it is:** A mathematical technique (STL / classical decomposition) that splits the time series into three separate components.

**How it works:** We use `statsmodels.tsa.seasonal.seasonal_decompose()` with `period=168` (168 hours = one week). This tells the algorithm to look for patterns that repeat every week.

**The three components extracted:**

| Component | What it captures | What we see in PJM |
|-----------|-----------------|-------------------|
| Trend | The long-term direction — is demand growing or shrinking over years? | Gradual rise from 2002–2010, then flattening. No sharp jumps. |
| Seasonality | The repeating pattern — what happens every week, predictably | A consistent weekly wave: demand rises Monday–Friday, dips Saturday–Sunday |
| Residual (noise) | Everything left over after removing trend and seasonality | Relatively small in PJM — the model is mostly trend + seasonality. Large in UCI — mostly residual. |

**What this tells us for the research:**
- In PJM, the residual (random part) is small relative to the seasonal and trend components. This means most of the variation in the data is systematic and learnable — which is why all 5 models achieve high R².
- In UCI, the residual dominates. Even after removing trend and weekly seasonality, most of the signal is unpredictable noise — explaining why R² never exceeds 0.57 on UCI.

---

### EDA Analysis 3 — Consumption Patterns (Hour / Day / Month)
**Chart saved:** `notebooks/figures/pjm_patterns.png` and `uci_patterns.png`

**What it is:** Three subplots showing average electricity demand grouped by (a) hour of day, (b) day of week, and (c) month of year.

**How it works:** We use `pandas groupby` to compute the mean demand for each group:
- `pjm_processed.groupby('hour')['PJME_MW'].mean()` — average demand for each of the 24 hours
- `pjm_processed.groupby('day_of_week')['PJME_MW'].mean()` — average for Monday to Sunday
- `pjm_processed.groupby('month')['PJME_MW'].mean()` — average for January to December

**What we see in PJM:**

*By hour of day:*
- Lowest demand: 4am–5am (~28,000 MW) — everyone asleep, minimal industrial activity
- First peak: 9am–10am (~40,000 MW) — offices and factories open
- Highest peak: 3pm–6pm (~45,000 MW) — afternoon air conditioning load + industrial activity
- Evening drop after 9pm as businesses close

*By day of week:*
- Monday to Friday: consistently higher demand (~42,000–44,000 MW average)
- Saturday: noticeably lower (~38,000 MW) — fewer industrial users
- Sunday: lowest (~36,000 MW) — most businesses and factories closed

*By month:*
- January–February: elevated (heating demand) — ~41,000 MW average
- March–May: lowest (mild weather, no heating or cooling needed) — ~36,000 MW
- June–August: highest (air conditioning) — peaking in July at ~46,000 MW
- September–December: gradually declining

**What we see in UCI:**
- Morning peak: 7am–9am (breakfast, appliances on)
- Evening peak: 7pm–10pm (family home, cooking, TV)
- Weekend patterns different to PJM: more usage at home on weekends, not less

**What this tells us for the research:**
- These systematic patterns are exactly what machine learning models exploit to make accurate predictions
- The fact that PJM has strong, consistent patterns by hour/day/month confirms that a 24-hour input window captures the most predictive information
- The time features added during preprocessing (hour, day_of_week, month, is_weekend, season) are directly justified by what this analysis reveals

---

### EDA Analysis 4 — Heatmap
**Chart saved:** `notebooks/figures/pjm_heatmap.png` and `uci_heatmap.png`

**What it is:** A colour-coded grid where each cell shows the average demand for one specific combination of hour and day of week.

**How it works:** We compute a pivot table — rows = 24 hours, columns = 7 days — and colour each cell from low (cool/blue) to high (warm/red) using a heatmap.

**What we see in PJM:**
- Darkest cells (highest demand): Tuesday–Thursday, 3pm–5pm — peak business hours mid-week
- Lightest cells (lowest demand): Sunday and Saturday, 3am–5am — weekend nights
- The gradient is smooth and consistent — no random bright spots — confirming the data is clean and patterned
- The weekday/weekend distinction is very clear in the colour bands

**What this tells us for the research:**
- The heatmap is a single image that immediately shows a supervisor or reader that this data has clear, learnable structure
- It visually validates that `is_weekend` and `hour` are important features
- It is used directly in the dashboard's Data Explorer page so readers can interact with it

---

### EDA Analysis 5 — Rolling Statistics
**Chart saved:** `notebooks/figures/pjm_rolling.png`

**What it is:** A plot of the 7-day rolling mean (smoothed demand) and rolling standard deviation (variability) over time.

**How it works:**
```
rolling_mean = pjm['PJME_MW'].rolling(window=168).mean()   # 7-day window (168 hours)
rolling_std  = pjm['PJME_MW'].rolling(window=168).std()
```

**What we see:**
- Rolling mean follows the same seasonal pattern as the raw time series — summer and winter peaks visible as a smooth wave
- Rolling standard deviation is higher in summer and winter (extreme weather causes more variable demand) and lower in spring/autumn
- No sudden jumps in the rolling mean — the series is stationary within each season, which is favourable for machine learning

**What this tells us for the research:**
- A stable rolling mean and manageable standard deviation confirm the data is suitable for supervised learning
- It supports the decision to use Min-Max scaling (the values fluctuate within a bounded and predictable range)

---

### EDA Analysis 6 — ACF / PACF (Autocorrelation Analysis)
**Chart saved:** `notebooks/figures/pjm_acf.png` and `uci_acf.png`

**What it is:** The most technically important EDA analysis. ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) measure how strongly the current hour's demand relates to previous hours.

**What "autocorrelation" means:** If I know electricity demand was 45,000 MW at 3pm yesterday, how useful is that for predicting demand at 3pm today? ACF quantifies this relationship at every possible time lag.

**How it works:**
```
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(pjm['PJME_MW'], lags=200)   # check lags up to 200 hours back
plot_pacf(pjm['PJME_MW'], lags=50)
```

**What the chart shows:**
- **Y-axis:** correlation value from -1 to +1. Values above the shaded band are statistically significant (not random).
- **X-axis:** the lag in hours

**Key spikes found in PJM:**

| Lag | Correlation | Meaning |
|-----|-------------|---------|
| Lag 1 | Very high (~0.95) | The previous hour's demand is a very strong predictor of the current hour |
| Lag 24 | High (~0.85) | The same hour yesterday is a strong predictor — daily cycle repeats |
| Lag 48 | Moderate | Two days ago is still relevant |
| Lag 168 | High (~0.75) | The same hour last week is a strong predictor — weekly cycle repeats |

**UCI results:** The same spikes exist but are weaker — lag 24 correlation is lower because individual human schedules vary more day to day.

**What this tells us for the research — this is the most critical EDA finding:**
- The spike at lag 24 directly justifies our **window size of 24 hours**. If we feed the model the last 24 hours, we are capturing the strongest predictive signal in the data.
- The spike at lag 168 shows a weekly pattern exists. A window of 168 would capture this too, but would make the model much larger and slower. We chose 24 as the minimum that captures the daily cycle.
- This is why the ACF/PACF chart is always cited when explaining the 24-hour window choice in the thesis.

---

### EDA Analysis 7 — UCI Correlation Matrix
**Chart saved:** `notebooks/figures/uci_correlation.png`

**What it is:** A heatmap showing how strongly each of the 7 UCI numerical columns correlates with each other. Only applies to UCI because PJM has only one numerical column.

**How it works:**
```
uci_numeric = uci_processed[['Global_active_power', 'Global_reactive_power',
                               'Voltage', 'Global_intensity',
                               'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
uci_numeric.corr()   # Pearson correlation coefficient between every pair
```

**What we see:**
- `Global_active_power` and `Global_intensity` are very strongly correlated (≈0.99) — they measure essentially the same thing (power = voltage × current)
- `Sub_metering_3` (water heater/AC) shows moderate correlation with overall power — it is the largest single electricity consumer
- `Voltage` shows slight negative correlation with power — when the house draws more current, voltage drops slightly (Ohm's law)
- `Sub_metering_1` and `Sub_metering_2` show low correlation — kitchen and laundry are used independently

**What this tells us for the research:**
- Confirms that `Global_active_power` is the right target variable (it is the most comprehensive measure of total consumption)
- Reveals that the sub-metering channels are largely independent — combining them would not necessarily help the model
- Justifies the decision to use only `Global_active_power` as the univariate input rather than all 7 columns together

---

### What EDA Produces — Files Saved

| File | Created by | What it shows |
|------|-----------|--------------|
| `pjm_timeseries.png` | Time series plot | 16 years of PJM demand |
| `pjm_decomposition.png` | Seasonal decomposition | Trend + seasonality + residual |
| `pjm_acf.png` | ACF/PACF | Lag correlations — justifies 24h window |
| `pjm_patterns.png` | Groupby plots | Average by hour / day / month |
| `pjm_heatmap.png` | Pivot heatmap | Hour vs day demand grid |
| `pjm_rolling.png` | Rolling statistics | 7-day rolling mean and std |
| `uci_timeseries.png` | Time series plot | 4 years of UCI household demand |
| `uci_decomposition.png` | Seasonal decomposition | UCI trend + seasonality + residual |
| `uci_acf.png` | ACF/PACF | UCI lag correlations |
| `uci_patterns.png` | Groupby plots | UCI average by hour / day / month |
| `uci_heatmap.png` | Pivot heatmap | UCI: hour vs day demand grid |
| `uci_correlation.png` | Correlation matrix | Relationships between UCI columns |

All 12 files are also documented in `notebooks/figures/figures_index.csv` with one-line descriptions.

---

## Step 3 — Model Training
### Notebook: `notebooks/04_models.ipynb`

**What is model training?**
We feed the historical data to each algorithm and let it learn the patterns.
Once trained, the model can predict future electricity usage it has never seen before.

---

### Input Variables — What Is Fed Into Every Model

All models receive the same input: a **24-hour sliding window** of normalised electricity demand values.

| Input | Description | Why used |
|-------|-------------|----------|
| `PJME_MW_scaled` (last 24 hours) | Normalised electricity demand values for the previous 24 hours (values between 0 and 1) | This is the core signal — past demand is the strongest predictor of future demand |

**Why 24 hours?**
- ACF/PACF analysis (Notebook 03) confirmed that electricity demand is most strongly correlated with the same hour the previous day (lag=24)
- 24 hours captures one full daily cycle (morning peak, afternoon peak, night trough)
- Industry standard for short-term electricity forecasting

**Input shape:**
- For tree models (RF, XGBoost): a flat array of 24 values `(24,)`
- For deep learning models (LSTM, GRU, CNN-LSTM): a 3D array `(24 timesteps, 1 feature)`

**Output (what the model predicts):**
- A single value: the electricity demand for the **next hour** (normalised, then converted back to MW)

---

### Why These 5 Models Were Chosen

| Model | Reason for selection |
|-------|---------------------|
| Random Forest | Strong baseline for tabular/time series regression; no gradient training required; handles non-linear patterns naturally |
| XGBoost | State-of-the-art gradient boosting; regularisation prevents overfitting; widely used in energy forecasting literature |
| LSTM | The most established deep learning method for time series; gating mechanism retains long-range memory |
| GRU | Simplified version of LSTM; fewer parameters; tests whether added complexity of LSTM is justified |
| CNN-LSTM | Hybrid model; CNN detects short-range local patterns, LSTM models how those patterns evolve — tests multi-scale learning |

Together they span the full spectrum: simple ensemble ML → boosted ML → recurrent DL → simplified recurrent DL → hybrid DL.

---

### Model 1 — Random Forest

**What it is:** An ensemble of 100 independent decision trees. Each tree is trained on a random subset of the data. Final prediction = average of all 100 trees.

**Why it works well for energy data:** Electricity demand follows clear rules (high in afternoon, low at night, high in summer). Decision trees capture these rule-based patterns very effectively.

**Parameters:**

| Parameter | Value | Why this value |
|-----------|-------|---------------|
| `n_estimators` | 100 | 100 trees gives stable predictions without excessive memory usage |
| `n_jobs` | -1 | Uses all available CPU cores for faster training |
| `random_state` | 42 | Fixed seed ensures reproducible results every run |

**No learning rate** — Random Forest does not use gradient descent. Each tree is built independently, not iteratively.

---

### Model 2 — XGBoost (Extreme Gradient Boosting)

**What it is:** Builds 200 trees one at a time. Each new tree focuses on correcting the mistakes of all previous trees. The final prediction is the sum of all 200 trees' contributions.

**Why it works well for energy data:** The boosting process (correcting errors step by step) is especially effective at capturing complex non-linear patterns in energy demand such as weather-driven spikes.

**Parameters:**

| Parameter | Value | Why this value |
|-----------|-------|---------------|
| `n_estimators` | 200 | More trees than RF because each tree is simpler; 200 gives good accuracy |
| `learning_rate` | 0.05 | Controls how much each new tree contributes; small value (0.05) means slow, careful learning which reduces overfitting |
| `max_depth` | 6 | Maximum depth of each tree; depth 6 allows complex patterns without memorising training data |
| `subsample` | 0.8 | Each tree uses only 80% of training rows (randomly chosen); prevents overfitting |
| `colsample_bytree` | 0.8 | Each tree uses only 80% of input features; adds randomness and improves generalisation |
| `random_state` | 42 | Fixed seed for reproducibility |

**Learning rate = 0.05** means: each boosting step moves 5% of the way toward correcting the error. Small steps → more stable → better final accuracy. If set too high (e.g. 0.5), the model overshoots and performance degrades.

---

### Model 3 — LSTM (Long Short-Term Memory)

**What it is:** A recurrent neural network with special memory cells. Unlike standard neural networks, LSTM processes the 24 input values one at a time in sequence, maintaining a hidden state (memory) that carries information forward through time.

**Why it works well for energy data:** Electricity demand has time-dependent patterns (what happened 3 hours ago affects now). LSTM's gating mechanism learns which past information to remember and which to forget.

**Architecture (layers in order):**

| Layer | Size | Purpose |
|-------|------|---------|
| LSTM layer 1 | 64 units, return_sequences=True | Processes the 24-hour sequence; outputs a sequence for the next layer |
| Dropout | 20% | Randomly switches off 20% of neurons during training to prevent memorisation |
| LSTM layer 2 | 32 units | Processes the output of layer 1; extracts higher-level temporal patterns |
| Dropout | 20% | Second regularisation layer |
| Dense (output) | 1 unit | Produces the final single prediction value |

**Training parameters:**

| Parameter | Value | Why this value |
|-----------|-------|---------------|
| Optimizer | Adam | Adaptive learning rate optimizer; most commonly used for neural networks |
| Initial learning rate | 0.001 (1e-3) | Standard starting point for Adam; balances speed and stability |
| Loss function | MSE (Mean Squared Error) | Penalises large prediction errors more heavily; appropriate for regression |
| Batch size | 512 | Processes 512 training samples per weight update; large batch = stable gradient estimates |
| Max epochs | 30 | Maximum of 30 full passes through the training data |
| Early Stopping patience | 5 | Stops training if validation loss does not improve for 5 consecutive epochs |
| ReduceLROnPlateau factor | 0.5 | Halves the learning rate when stuck; patience=3 epochs |

**What the learning rate does:** Controls how big each step is when adjusting the model's internal weights. Too high = overshoots the optimal solution. Too low = takes forever to train. Starting at 0.001 and halving when stuck is a proven strategy.

---

### Model 4 — GRU (Gated Recurrent Unit)

**What it is:** A simplified version of LSTM with fewer internal gates (2 gates instead of 3). Slightly faster to train with comparable accuracy.

**Why included:** To test whether LSTM's extra complexity is justified or whether a simpler recurrent model achieves the same result.

**Architecture:**

| Layer | Size | Purpose |
|-------|------|---------|
| GRU layer 1 | 64 units, return_sequences=True | Processes the 24-hour sequence |
| Dropout | 20% | Regularisation |
| GRU layer 2 | 32 units | Higher-level temporal patterns |
| Dropout | 20% | Regularisation |
| Dense (output) | 1 unit | Final prediction |

**All training parameters are identical to LSTM** (same optimizer, learning rate, batch size, epochs, callbacks) so the comparison is fair — the only difference is the cell type.

---

### Model 5 — CNN-LSTM (Hybrid)

**What it is:** Combines a Convolutional Neural Network (CNN) with an LSTM. The CNN first scans the 24-hour input to detect local patterns (e.g. a sharp rise over 3 hours). The LSTM then models how those detected patterns evolve.

**Why included:** Tests whether extracting local features first (CNN) before modelling temporal dependencies (LSTM) improves forecasting accuracy over LSTM alone.

**Architecture:**

| Layer | Size | Purpose |
|-------|------|---------|
| Conv1D layer 1 | 64 filters, kernel size=3, ReLU activation | Scans 3-hour windows across the 24-hour input to detect local patterns |
| MaxPooling1D | pool size=2 | Reduces the sequence length by half; keeps only the strongest features |
| Conv1D layer 2 | 32 filters, kernel size=3, ReLU activation | Detects higher-level combinations of the patterns found by layer 1 |
| LSTM | 50 units | Models how the CNN-extracted patterns change over time |
| Dropout | 20% | Regularisation |
| Dense (output) | 1 unit | Final prediction |

**All training parameters are identical to LSTM and GRU** for a fair comparison.

---

### Training Settings Summary (All Deep Learning Models)

| Setting | Value | Reason |
|---------|-------|--------|
| Optimizer | Adam | Adaptive, self-tuning, widely used standard |
| Initial learning rate | 0.001 | Standard starting point; neither too fast nor too slow |
| Loss function | Mean Squared Error (MSE) | Appropriate for continuous value regression; large errors penalised more |
| Batch size | 512 | Large enough for stable gradients; fits comfortably in memory |
| Maximum epochs | 30 | Upper limit; Early Stopping usually kicks in before this |
| Early Stopping | patience=5, monitor val_loss | Stops when validation loss stops improving; prevents overfitting |
| ReduceLROnPlateau | factor=0.5, patience=3 | Halves learning rate after 3 epochs of no improvement |
| Validation split | 10% of training data | Monitors generalisation during training; also chronological |
| Train/Test split | 80% train / 20% test | Standard split; chronological (no shuffling) to prevent data leakage |
| Window size | 24 hours | One full daily cycle; justified by ACF analysis |
| Input normalisation | Min-Max scaling to [0,1] | Required for neural networks; tree models also benefit |

**Files saved after this step:**
- `models/rf_pjm.pkl` — saved Random Forest model
- `models/xgb_pjm.json` — saved XGBoost model
- `models/lstm_pjm.keras` — saved LSTM model
- `models/gru_pjm.keras` — saved GRU model
- `models/cnnlstm_pjm.keras` — saved CNN-LSTM model
- `models/all_predictions.pkl` — every model's predictions on the test set saved together

---

## Step 4 — Evaluation
### Notebook: `notebooks/05_evaluation.ipynb`

**What is evaluation?**
We measure how accurate each model's predictions are on data it has never seen (the test set).

**The 4 metrics used:**

### RMSE — Root Mean Squared Error
- Measures the average size of prediction errors in megawatts
- Squaring the errors means large mistakes are punished more heavily
- Lower is better
- Example: RMSE=387 means the model is wrong by about 387 MW on average

### MAE — Mean Absolute Error
- Simply the average of how wrong each prediction is (in MW)
- Less sensitive to occasional large errors than RMSE
- Lower is better
- Example: MAE=270 means on average the prediction is off by 270 MW

### MAPE — Mean Absolute Percentage Error
- The error expressed as a percentage of the actual value
- Useful for comparing across different scales
- Lower is better
- Example: MAPE=0.87% means predictions are off by less than 1% on average

### R² — Coefficient of Determination
- Measures how much of the variation in electricity usage the model can explain
- Ranges from 0 to 1 (higher is better)
- R²=0.9962 means the model explains 99.62% of all variation in the data
- Anything above 0.95 is considered excellent for energy forecasting

**Files saved after this step:**
- `notebooks/figures/all_metrics.xlsx` — full results table in Excel format
- `notebooks/figures/pjm_metrics.csv` — results in CSV format
- `notebooks/figures/pjm_actual_vs_pred.png` — chart of actual vs predicted values
- `notebooks/figures/pjm_comparison.png` — bar chart comparing all models
- `notebooks/figures/pjm_radar.png` — radar chart showing strengths of each model

---

## Final Results — PJM East Dataset

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
|-------|-----------|----------|----------|----|
| **Random Forest** | **387.94** | **270.74** | **0.87** | **0.9962** |
| XGBoost | 436.57 | 325.77 | 1.06 | 0.9952 |
| LSTM | 536.77 | 412.25 | 1.35 | 0.9928 |
| GRU | 1,755.82 | 1,415.41 | 4.74 | 0.9227 |
| CNN-LSTM | 1,868.88 | 1,491.27 | 4.93 | 0.9124 |

**Bold = best value in each column**

---

## What the Results Mean

- **Random Forest is the best model** — it wins on all four metrics
- **XGBoost is a close second** — only slightly worse than Random Forest
- **LSTM performs reasonably** — acceptable accuracy but worse than tree models
- **GRU and CNN-LSTM struggled** — significantly worse; the 24-hour window was not long enough for these complex models to learn effectively
- **Key finding:** Traditional machine learning (Random Forest, XGBoost) outperforms deep learning (LSTM, GRU, CNN-LSTM) for short-term energy forecasting with a 24-hour look-back window

---

## Understanding the Results Files

After running all four notebooks, the results are saved in several files. This section explains exactly what each file contains and how to read the numbers.

---

### File: `notebooks/figures/pjm_metrics.csv`
**What it is:** The final performance results for all 5 models tested on the PJM East dataset. This is the primary results file for the thesis.

**What it looks like when you open it in Excel:**

| Model | RMSE | MAE | MAPE (%) | R² |
|-------|------|-----|----------|----|
| Random Forest | 387.94 | 270.74 | 0.87 | 0.9962 |
| XGBoost | 436.57 | 325.77 | 1.06 | 0.9952 |
| LSTM | 536.77 | 412.25 | 1.35 | 0.9928 |
| GRU | 1755.82 | 1415.41 | 4.74 | 0.9227 |
| CNN-LSTM | 1868.88 | 1491.27 | 4.93 | 0.9124 |

**What each column means:**

| Column | Full name | Unit | How to read it | Better when |
|--------|-----------|------|---------------|-------------|
| `RMSE` | Root Mean Squared Error | Megawatts (MW) | On average, the model's prediction is off by this many MW | Lower is better |
| `MAE` | Mean Absolute Error | Megawatts (MW) | On average, the absolute error per hour is this many MW | Lower is better |
| `MAPE (%)` | Mean Absolute Percentage Error | Percentage (%) | On average, the error is this percentage of the actual value | Lower is better |
| `R²` | R-squared (coefficient of determination) | Unitless (0 to 1) | How much of the variation in actual demand the model explains | Higher is better (1.0 = perfect) |

**Plain English interpretation of the PJM results:**
- Random Forest RMSE = 387.94 MW means: on average, its prediction is off by about 388 MW out of a typical demand of 45,000 MW — that is less than 1% error
- Random Forest R² = 0.9962 means: the model explains 99.62% of all variation in electricity demand — this is excellent
- GRU RMSE = 1755.82 MW means: its prediction is off by 1,756 MW on average — about 4 times worse than Random Forest
- MAPE = 0.87% for Random Forest means: for any given hour, the prediction is within 0.87% of the real value on average

---

### File: `notebooks/figures/uci_metrics.csv`
**What it is:** The same results table but for the UCI household dataset. Same column structure as above.

| Model | RMSE | MAE | MAPE (%) | R² |
|-------|------|-----|----------|----|
| Random Forest | 0.4778 | 0.3339 | 44.36 | 0.5562 |
| XGBoost | 0.4728 | 0.3302 | 44.47 | 0.5655 |
| LSTM | 0.6848 | 0.5487 | 82.00 | 0.0881 |
| GRU | 0.5931 | 0.4541 | 67.07 | 0.3160 |
| CNN-LSTM | 0.6818 | 0.5542 | 86.06 | 0.0962 |

**Why these numbers look so different from PJM:**
- RMSE is in kilowatts (kW) not megawatts (MW) — household power is much smaller in scale
- MAPE of 44% looks alarming but is expected: when a house consumes 0.1 kW (almost nothing switched on), any small prediction error becomes a huge percentage
- R² of 0.55 for the best model means: even the best model can only explain 55% of household consumption variation — individual behaviour is simply too unpredictable
- **This is not a failure.** It is a finding: household-level data is harder to forecast than grid-level data, and this research demonstrates why

---

### File: `notebooks/figures/all_metrics.xlsx`
**What it is:** An Excel file with both datasets' results in one place. Use this file when writing the thesis report.

**It contains two sheets:**
- Sheet 1: `PJM` — same data as pjm_metrics.csv
- Sheet 2: `UCI` — same data as uci_metrics.csv

**How to open it:** Double-click in Windows. It will open in Microsoft Excel.

---

### File: `notebooks/figures/figures_index.csv`
**What it is:** A reference guide to every chart image saved in the `notebooks/figures/` folder. Open it in Excel to get a one-line description of what each figure shows.

**Columns:**
- `Figure` — the name of the chart
- `Path` — the file path to find the image
- `Description` — one sentence explaining what the chart shows

**How to use it:** If you open `notebooks/figures/` in Windows Explorer and see `pjm_radar.png` but don't know what it is, check this CSV — the description will tell you without opening the file.

---

### How to Tell Which Model Won — Quick Reference

| Metric | PJM winner | UCI winner | What low/high means |
|--------|-----------|-----------|---------------------|
| RMSE | Random Forest (388 MW) | XGBoost (0.473 kW) | Lower = fewer MW/kW error on average |
| MAE | Random Forest (271 MW) | XGBoost (0.330 kW) | Lower = smaller typical mistake |
| MAPE | Random Forest (0.87%) | Random Forest (44.4%) | Lower = smaller % error relative to actual |
| R² | Random Forest (0.9962) | XGBoost (0.5655) | Closer to 1.0 = better fit to real data |

**Summary in one sentence:** Random Forest is the best model for regional grid data (PJM). XGBoost is marginally better for household data (UCI). All deep learning models underperform both tree models in this research.

---

## The Dashboard
### File: `dashboard/app.py`

- An interactive web application built with Streamlit
- Loads all results and displays them visually in a browser
- Pages available:
  - **Home** — summary of datasets and models
  - **Data Explorer** — explore time series, patterns, and heatmaps interactively
  - **Forecasts** — compare actual vs predicted values for any model
  - **Performance** — view metrics table, bar charts, radar chart, and training curves
  - **Research Results** — cross-dataset analysis and statistical tests
- Run it with: `streamlit run app.py`
- Open browser at: http://localhost:8501

---

## Summary of All Files

### Raw Data (do not modify these)

| File | Size | What it contains |
|------|------|-----------------|
| `data/raw/PJME_hourly.csv` | ~5 MB | 145,366 rows, 2 columns: Datetime + PJME_MW. Primary dataset. |
| `data/raw/household_power_consumption.txt` | 127 MB | ~2 million rows, 9 columns. UCI household dataset. |

### Processed Data (created by Notebook 02)

| File | Format | What it contains |
|------|--------|-----------------|
| `data/processed/pjm_processed.csv` | CSV (Excel-readable) | Cleaned PJM data with 10 columns including time features and scaled values |
| `data/processed/uci_processed.csv` | CSV (Excel-readable) | Cleaned UCI data with 16 columns including all sub-metering channels |
| `data/processed/pjm_sequences.npz` | NumPy binary (not Excel) | Arrays: X_train, y_train, X_test, y_test for PJM — ready for model input |
| `data/processed/uci_sequences.npz` | NumPy binary (not Excel) | Same arrays for UCI dataset |
| `data/processed/pjm_scaler.pkl` | Python binary | Min-Max scaler settings for PJM — required to convert predictions back to MW |
| `data/processed/uci_scaler.pkl` | Python binary | Min-Max scaler settings for UCI — required to convert predictions back to kW |

### Notebooks (run in order)

| File | Runtime | What it does |
|------|---------|-------------|
| `notebooks/02_preprocessing.ipynb` | 2–3 min | Cleans data, creates sequences, saves processed files |
| `notebooks/03_eda.ipynb` | 3–5 min | Creates all visualisation charts, saves to figures/ |
| `notebooks/04_models.ipynb` | 60–90 min | Trains all 5 models, saves predictions |
| `notebooks/05_evaluation.ipynb` | 2–3 min | Calculates metrics, saves results tables |

### Results Files (use these in the report)

| File | Format | What it contains |
|------|--------|-----------------|
| `notebooks/figures/pjm_metrics.csv` | CSV (Excel-readable) | 5 rows × 5 columns: Model, RMSE, MAE, MAPE, R² for PJM |
| `notebooks/figures/uci_metrics.csv` | CSV (Excel-readable) | Same structure for UCI dataset |
| `notebooks/figures/all_metrics.xlsx` | Excel | Both datasets' results in one file — two sheets |
| `notebooks/figures/figures_index.csv` | CSV (Excel-readable) | Index of all chart images with one-line descriptions |

### Chart Images (created by Notebooks 03 and 05)

| File | What it shows |
|------|--------------|
| `pjm_timeseries.png` | Full 16-year PJM demand line |
| `pjm_decomposition.png` | Trend + seasonality + noise separated |
| `pjm_acf.png` | Lag correlation — justifies 24-hour window choice |
| `pjm_patterns.png` | Average usage by hour / day / month |
| `pjm_heatmap.png` | Hour vs day grid — when demand is highest |
| `pjm_rolling.png` | Rolling average over time |
| `pjm_actual_vs_pred.png` | Actual demand vs each model's prediction |
| `pjm_comparison.png` | Bar chart: all 5 models on all 4 metrics |
| `pjm_radar.png` | Radar chart: all metrics in one shape per model |
| `dl_training_curves.png` | LSTM/GRU/CNN-LSTM training loss over epochs |
| `uci_timeseries.png` | Full 4-year UCI household demand line |
| `uci_decomposition.png` | UCI trend + seasonality + noise |
| `uci_acf.png` | UCI lag correlations |
| `uci_patterns.png` | Average UCI usage by hour / day / month |
| `uci_heatmap.png` | UCI: hour vs day demand grid |
| `uci_correlation.png` | Correlation between UCI sub-metering channels |
| `uci_actual_vs_pred.png` | UCI actual vs each model's prediction |
| `uci_comparison.png` | UCI bar chart across all 5 models |
| `uci_radar.png` | UCI radar chart |

### Model Files

| File | Size | What it contains |
|------|------|-----------------|
| `models/rf_pjm.pkl` | ~980 MB | Trained Random Forest model (best performing — keep this) |
| `models/all_predictions.pkl` | Small | All 5 models' predictions for both datasets — used by Notebook 05 |

### Other Files

| File | What it is |
|------|-----------|
| `dashboard/app.py` | Interactive web dashboard — run with `streamlit run app.py` |
| `requirements.txt` | List of Python packages to install |
| `README.md` | Windows step-by-step setup and run guide |
| `IMPLEMENTATION_GUIDE.md` | This file — full explanation of the project |
| `notebooks/figures/figures_index.csv` | Index of all chart images with descriptions |

---

## Checklist — Files to Open When Writing the Report

| Question | Open this file |
|----------|---------------|
| What are the PJM results? | `notebooks/figures/pjm_metrics.csv` |
| What are the UCI results? | `notebooks/figures/uci_metrics.csv` |
| Both results together? | `notebooks/figures/all_metrics.xlsx` |
| What does chart X show? | `notebooks/figures/figures_index.csv` |
| What model parameters were used? | Section "Model Parameters, Input Variables and Design Decisions" in this file |
| What does the raw data look like? | `data/raw/PJME_hourly.csv` — open in Excel |
| What does the processed data look like? | `data/processed/pjm_processed.csv` — open in Excel |

---

*COM748 Masters Research · 2025–2026*
