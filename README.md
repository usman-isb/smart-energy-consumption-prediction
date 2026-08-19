# Smart Energy Consumption Forecasting

---

## How to Run This Project on a Windows Machine (Step by Step)

Follow every step in order. Do not skip any step.

---

## STEP 1 — Install Python

1. Open your browser and go to: **https://www.python.org/downloads/**
2. Click the yellow **"Download Python 3.x.x"** button
3. Open the downloaded file
4. **IMPORTANT:** On the first screen, tick the box that says **"Add Python to PATH"**
5. Click **"Install Now"**
6. Wait for installation to finish, then click **Close**

To verify Python is installed correctly:
- Press `Win + R`, type `cmd`, press Enter
- Type the following and press Enter:
```
python --version
```
You should see something like `Python 3.11.x`. If you see an error, reinstall Python and make sure to tick "Add to PATH".

---

## STEP 2 — Download or Copy This Project

Place the entire project folder on your computer. For example:
```
C:\Users\YourName\Desktop\implementation
```

Make sure the folder contains:
- `data/` folder
- `notebooks/` folder
- `dashboard/` folder
- `requirements.txt`
- `README.md`

---

## STEP 2b — Download the UCI Dataset (Required)

One dataset file is **not included** in this project because it is 127 MB (too large for GitHub).
You must download it separately before running any notebooks.

**Option A — Automatic download script (recommended):**

After completing Step 6 (activating the virtual environment), run:
```
python download_data.py
```
This will automatically download and save the file to the correct folder.
It takes 2–10 minutes depending on your internet speed.

**Option B — Manual download:**

1. Open your browser and go to:
   **https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption**
2. Click the **Download** button
3. Extract the downloaded zip file
4. Copy the file named `household_power_consumption.txt` into:
   ```
   C:\Users\YourName\Desktop\implementation\data\raw\
   ```
5. Rename a copy of it to `household_power_consumption.csv` (keep both files in the same folder)

**You will know it is ready when** you can see both of these files:
```
data\raw\household_power_consumption.txt
data\raw\household_power_consumption.csv
```

**Note:** The PJM East dataset (`PJME_hourly.csv`) is already included in the project folder — you do not need to download it.

---

## STEP 3 — Open Command Prompt

1. Press `Win + R`
2. Type `cmd`
3. Press **Enter**

A black window will open. This is the Command Prompt.

---

## STEP 4 — Navigate to the Project Folder

In Command Prompt, type the following (replace the path with where you saved the project):

```
cd "C:\Users\YourName\Desktop\implementation"
```

Press **Enter**.

To confirm you are in the right folder, type:
```
dir
```
Press Enter. You should see files like `requirements.txt` and folders like `data`, `notebooks`, `dashboard`.

---

## STEP 5 — Create a Virtual Environment

A virtual environment keeps all the project packages separate from your system Python.

Type the following and press Enter:
```
python -m venv venv
```

Wait for it to finish. A new folder called `venv` will appear inside the project folder.

---

## STEP 6 — Activate the Virtual Environment

Type the following and press Enter:
```
venv\Scripts\activate
```

You will see `(venv)` appear at the beginning of the line, like this:
```
(venv) C:\Users\YourName\Desktop\implementation>
```

**Important:** You must see `(venv)` before running any further commands. If you close Command Prompt and reopen it, you must run this step again.

---

## STEP 7 — Install Required Packages

Type the following and press Enter:
```
pip install -r requirements.txt
```

This will download and install all required packages. It will take **5 to 15 minutes** depending on your internet speed. You will see a lot of text scrolling — this is normal. Wait until it finishes and you see the command prompt again.

---

## STEP 8 — Register the Jupyter Kernel

This links the virtual environment to Jupyter Notebook so it can find all the installed packages.

Type the following and press Enter:
```
python -m ipykernel install --user --name=thesis-venv --display-name "Python 3 (thesis-venv)"
```

You should see a message like:
```
Installed kernelspec thesis-venv in C:\Users\YourName\...
```

---

## STEP 9 — Launch Jupyter Notebook

Type the following and press Enter:
```
jupyter notebook
```

A browser window will open automatically showing the Jupyter file browser.

If the browser does not open automatically:
- Look at the Command Prompt window
- Find a line that starts with `http://localhost:8888/...`
- Copy that entire link and paste it into your browser

---

## STEP 10 — Run the Notebooks in Order

In the Jupyter browser, click on the `notebooks` folder. Run the notebooks **in the following order**. Do not skip any.

---

### Notebook 1 of 4 — Data Preprocessing

**File:** `02_preprocessing.ipynb`

**What it does:**
- Loads the two raw datasets (PJM and UCI)
- Cleans missing values and removes duplicate timestamps
- Adds time features (hour, day, month, season, etc.)
- Normalises the data between 0 and 1
- Creates 24-hour sliding window sequences for model input
- Saves all processed files to the `data/processed/` folder

**How to run:**
1. Click on `02_preprocessing.ipynb` to open it
2. If asked to select a kernel, choose **"Python 3 (thesis-venv)"**
3. Click **Kernel** in the top menu
4. Click **Restart & Run All**
5. Click **Restart and Run All Cells** when the confirmation box appears
6. Wait until all cells finish running. A number like `[12]` will appear next to each cell when it is done.

**Expected time:** 2–3 minutes

**You will know it worked when** you see a `processed` folder appear inside `data/`.

---

### Notebook 2 of 4 — Exploratory Data Analysis

**File:** `03_eda.ipynb`

**What it does:**
- Plots time series graphs for both datasets
- Shows seasonal patterns (by hour, day, month, season)
- Performs autocorrelation analysis (ACF/PACF)
- Creates heatmaps
- Saves all charts to `notebooks/figures/`

**How to run:**
1. Go back to the Jupyter file browser
2. Click on `03_eda.ipynb`
3. Click **Kernel → Restart & Run All**
4. Wait for all cells to complete

**Expected time:** 3–5 minutes

**You will know it worked when** you see image files appear in `notebooks/figures/`.

---

### Notebook 3 of 4 — Model Training

**File:** `04_models.ipynb`

**What it does:**
- Trains all 5 models on both datasets (10 models total):
  - Random Forest
  - XGBoost
  - LSTM (deep learning)
  - GRU (deep learning)
  - CNN-LSTM (deep learning)
- Saves all trained models to the `models/` folder
- Saves all predictions to `models/all_predictions.pkl`

**How to run:**
1. Click on `04_models.ipynb`
2. Click **Kernel → Restart & Run All**
3. Wait — this notebook takes a long time

**Expected time:** 60 to 90 minutes

**Do not close the browser or the Command Prompt window while this is running.**

**You will know it worked when** you see `.keras` and `.json` and `.pkl` files appear in the `models/` folder.

---

### Notebook 4 of 4 — Evaluation

**File:** `05_evaluation.ipynb`

**What it does:**
- Loads all predictions from `models/all_predictions.pkl`
- Calculates performance metrics: RMSE, MAE, MAPE, R²
- Creates comparison charts and radar charts
- Saves results to `notebooks/figures/all_metrics.xlsx`

**How to run:**
1. Click on `05_evaluation.ipynb`
2. Click **Kernel → Restart & Run All**
3. Wait for all cells to complete

**Expected time:** 2–3 minutes

---

## STEP 11 — Run the Research Dashboard (Optional)

The research dashboard lets you explore all results visually in your browser.

1. Open a **new** Command Prompt window (`Win + R` → `cmd`)
2. Navigate to the project folder:
```
cd "C:\Users\YourName\Desktop\implementation"
```
3. Activate the virtual environment:
```
venv\Scripts\activate
```
4. Navigate to the dashboard folder:
```
cd dashboard
```
5. Run the dashboard:
```
streamlit run app.py
```
6. Open your browser and go to: **http://localhost:8501**

**To stop the dashboard:** Press `Ctrl + C` in the Command Prompt window.

---

## STEP 12 — Generate the Prediction Model

This step trains the best model (Random Forest) on its own and saves it as a single file.
This is required before you can run the prediction web application in Step 13.

**Note:** If you already ran Notebook 3 of 4 (`04_models.ipynb`) in Step 10, the model file
`models/rf_pjm.pkl` already exists and you can skip this step and go straight to Step 13.

**How to check:** Open the `models/` folder inside the project. If you see `rf_pjm.pkl`, skip to Step 13.

**If the model file is missing, run Notebook 06:**

1. In the Jupyter browser, click on the `notebooks` folder
2. Click on `06_best_model.ipynb`
3. If asked to select a kernel, choose **"Python 3 (thesis-venv)"**
4. Click **Kernel → Restart & Run All**
5. Click **Restart and Run All Cells** when the confirmation box appears
6. Wait for all cells to finish

**What this notebook does:**
- Loads the raw PJM East dataset
- Cleans and normalises the data
- Creates 24-hour input sequences
- Trains the Random Forest model
- Saves three files needed by the web app:
  - `models/rf_pjm.pkl` — the trained model
  - `data/processed/pjm_scaler.pkl` — scaling settings
  - `data/processed/pjm_processed.csv` — processed data for input lookup

**Expected time:** 20–30 minutes

**You will know it worked when** you see `rf_pjm.pkl` appear in the `models/` folder.

---

## STEP 13 — Run the Prediction Web Application

The prediction web app lets you interactively predict electricity consumption.
It has two modes:

- **Historical Prediction** — pick any date from 2015–2018 and see the model predict that hour in real time
- **Future Forecast** — pick a future year, month, and day type to see a full 24-hour demand forecast based on historical patterns

**Requirement:** Complete Step 12 first (the model file must exist).

**How to run:**

1. Open a **new** Command Prompt window (`Win + R` → `cmd`)
2. Navigate to the project folder:
```
cd "C:\Users\YourName\Desktop\implementation"
```
3. Activate the virtual environment:
```
venv\Scripts\activate
```
4. Navigate to the prediction app folder:
```
cd predict_app
```
5. Run the web app:
```
streamlit run app.py
```
6. Open your browser and go to: **http://localhost:8502**

**To stop the app:** Press `Ctrl + C` in the Command Prompt window.

---

### How to Use the Prediction App

**Tab 1 — Historical Prediction:**
1. Select any date between 2015 and 2018 using the date picker
2. Choose the hour of the day using the slider (0 = midnight, 12 = noon, 23 = 11 PM)
3. Click **Predict**
4. The app shows:
   - **Predicted MW** — what the model forecasted
   - **Actual MW** — what really happened
   - **Error** — how far off the prediction was
   - **A chart** — 24-hour input window with the prediction (orange star) vs actual (green dot)

**Tab 2 — Future Forecast:**
1. Select a future year (2019 or later)
2. Select a month
3. Select a day type (Monday to Sunday)
4. Click **Generate Forecast**
5. The app shows:
   - A full **24-hour demand curve** for that type of day
   - The **historical range** from 2002–2018 as a shaded band
   - The model's **hour-by-hour prediction** as an orange line
   - Peak demand hour, lowest demand hour, and daily average

**Note:** The Future Forecast is based on historical patterns from 2002–2018.
It shows what a typical day of that type would look like — it does not predict specific future events.

---

## Final Results Summary

After running all four notebooks, these are the results:

### PJM East — Regional Grid

| Model | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
|-------|-----------|----------|----------|----|
| **Random Forest** | **387.94** | **270.74** | **0.87** | **0.9962** |
| XGBoost | 436.57 | 325.77 | 1.06 | 0.9952 |
| LSTM | 536.77 | 412.25 | 1.35 | 0.9928 |
| GRU | 1,755.82 | 1,415.41 | 4.74 | 0.9227 |
| CNN-LSTM | 1,868.88 | 1,491.27 | 4.93 | 0.9124 |

### UCI Household

| Model | RMSE (kW) | MAE (kW) | MAPE (%) | R² |
|-------|-----------|----------|----------|----|
| Random Forest | 0.4778 | 0.3339 | 44.36 | 0.5562 |
| **XGBoost** | **0.4728** | **0.3302** | **44.47** | **0.5655** |
| LSTM | 0.6848 | 0.5487 | 82.00 | 0.0881 |
| GRU | 0.5931 | 0.4541 | 67.07 | 0.3160 |
| CNN-LSTM | 0.6818 | 0.5542 | 86.06 | 0.0962 |

**Key finding:** Random Forest and XGBoost outperform all deep learning models on both datasets with a 24-hour look-back window.

---

## Model Parameters

| Model | Parameters |
|-------|------------|
| Random Forest | 100 trees, all CPU cores, random_state=42 |
| XGBoost | 200 trees, learning_rate=0.05, max_depth=6, subsample=0.8 |
| LSTM | LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1) |
| GRU | GRU(64) → Dropout(0.2) → GRU(32) → Dropout(0.2) → Dense(1) |
| CNN-LSTM | Conv1D(64, k=3) → MaxPool → Conv1D(32, k=3) → LSTM(50) → Dropout → Dense(1) |
| All DL models | Adam lr=0.001, MSE loss, batch=512, max 30 epochs, EarlyStopping patience=5 |
| Window size | 24 hours |
| Train / Test split | 80% / 20% (chronological, no shuffling) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` is not recognised | Reinstall Python and tick "Add to PATH" |
| `venv\Scripts\activate` gives an error | Open PowerShell as Administrator and run: `Set-ExecutionPolicy RemoteSigned` |
| Packages fail to install | Try: `pip install -r requirements.txt --user` |
| Kernel not found in Jupyter | Re-run Step 8, then refresh the Jupyter browser page |
| Notebook 04 is very slow | This is normal. Random Forest takes 20–30 min on PJM. Do not close the window. |
| Computer runs out of memory | Close all other programs before running Notebook 04 |
| Streamlit shows blank page | Make sure Notebooks 04 and 05 have been fully run first |
| `(venv)` not showing | You need to activate the environment again (Step 6) |

---

*2025–2026*
