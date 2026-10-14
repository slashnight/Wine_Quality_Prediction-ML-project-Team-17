# Wine Quality Studio UI

An upgraded, interactive Streamlit application for analyzing and predicting wine quality using machine learning models trained on the benchmark UCI Wine Quality dataset.

---

## ✨ Features & Capabilities

### 1. Dataset Exploration & Batch Prediction
- **Direct Dataset Download:** Download the complete raw UCI Wine Quality dataset (6,497 bottles, 11 chemical attributes) as a CSV file.
- **Dynamic Bottle Count Input & Real-Time Validation:** 
  - Allows the user to specify any bottle sample size.
  - Automatically calculates the bottle count and checks that it **does not exceed the total available bottles** in the dataset (6,497).
  - Displays instant visual confirmation when valid, or a warning banner preventing invalid runs if exceeded.
- **Dual ML Models Execution:**
  - **Model 1: Binary Logistic Regression:** Classifies wine as **Good** (sensory score $\ge 7$) or **Bad** (score $\le 6$) with prediction confidence / probability %.
  - **Model 2: Multinomial Logistic Regression:** Predicts the exact sensory quality rating score on a scale from 3 to 9.
- **Dataset Prediction Dashboard:** KPI metrics (Total bottles evaluated, Good vs Bad percentages, Average rating), bottle-by-bottle preview table, and CSV download of prediction results.
- **Dedicated Separate Graph Section:** Placed directly below the prediction dashboard:
  - Quality Rating Distribution Bar Chart
  - Good vs. Bad Classification Proportion Donut Chart
  - Key Chemical Attributes Comparison Grouped Bar Chart (comparing alcohol, volatile acidity, citric acid, and sulphates)

### 2. Customized Wine Data Prediction (11 Attributes Configurator)
- **Interactive Option Toggle:** Dedicated button / toggle to reveal the customized prediction workstation.
- **11 Physicochemical Attributes Customization:**
  - Modify and access all 11 chemical attributes:
    1. Fixed Acidity ($g/dm^3$)
    2. Volatile Acidity ($g/dm^3$)
    3. Citric Acid ($g/dm^3$)
    4. Residual Sugar ($g/dm^3$)
    5. Chlorides ($g/dm^3$)
    6. Free Sulfur Dioxide ($mg/dm^3$)
    7. Total Sulfur Dioxide ($mg/dm^3$)
    8. Density ($g/cm^3$)
    9. pH
    10. Sulphates ($g/dm^3$)
    11. Alcohol (% vol)
- **Quick Preset Selectors:** One-click presets for High-Quality Reserve, Dataset Average, High Volatile Acidity (Flawed), or Reset to Default.
- **Customized Prediction Dashboard:** High-contrast Good/Bad verdict banner, quality score, confidence level, and CSV export.
- **Dedicated Separate Graph Section:** Placed directly below the customized prediction dashboard:
  - **Interactive Radar / Spider Chart:** Compares your custom sample against benchmark dataset averages for Good wines and Bad wines.
  - **Probability Gauge Speedometer:** Visualizes the model's Good quality probability.
  - **Chemical Profile Horizontal Bar Chart:** Shows each entered attribute value.

### 3. Rock-Solid Button & State Management
- Seamless Streamlit session state architecture ensuring that clicking download buttons or editing values does not clear results or break button flows.
- Dedicated "Clear / Reset" controls for both prediction tasks.

---

## 🚀 How to Run Locally

1. Open PowerShell or a terminal in the project directory:
   ```bash
   cd c:\Users\ADMIN\Desktop\wine_quality_ui
   ```

2. (Optional) Activate your Python virtual environment if configured.

3. Install requirements (if not already installed):
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Open the displayed local URL in your web browser (typically `http://localhost:8501`).
