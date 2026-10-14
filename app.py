import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Studio | ML Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Constants & Attribute Definitions
# ---------------------------------------------------------
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

DISPLAY_NAMES = {
    "fixed_acidity": "Fixed Acidity",
    "volatile_acidity": "Volatile Acidity",
    "citric_acid": "Citric Acid",
    "residual_sugar": "Residual Sugar",
    "chlorides": "Chlorides",
    "free_sulfur_dioxide": "Free Sulfur Dioxide",
    "total_sulfur_dioxide": "Total Sulfur Dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol",
}

UNITS = {
    "fixed_acidity": "g/dm³ (tartaric acid)",
    "volatile_acidity": "g/dm³ (acetic acid)",
    "citric_acid": "g/dm³",
    "residual_sugar": "g/dm³",
    "chlorides": "g/dm³ (sodium chloride)",
    "free_sulfur_dioxide": "mg/dm³",
    "total_sulfur_dioxide": "mg/dm³",
    "density": "g/cm³",
    "pH": "pH scale",
    "sulphates": "g/dm³ (potassium sulphate)",
    "alcohol": "% by volume",
}

HELP_TEXT = {
    "fixed_acidity": "Fixed non-volatile acids contributing to wine freshness and crispness.",
    "volatile_acidity": "Volatile acids (primarily acetic acid); excessive levels lead to an unpleasant vinegar taste.",
    "citric_acid": "Small amounts add 'freshness' and flavor to wine.",
    "residual_sugar": "Amount of natural grape sugar remaining after fermentation ceases.",
    "chlorides": "Amount of salt in the wine.",
    "free_sulfur_dioxide": "Free form of SO2 preventing microbial growth and wine oxidation.",
    "total_sulfur_dioxide": "Sum of free and bound SO2 molecules in the wine sample.",
    "density": "Density of the wine liquid, closely influenced by alcohol and sugar percentage.",
    "pH": "Acidity / alkalinity scale (most wines rank between 3.0 and 4.0).",
    "sulphates": "Wine additive that acts as an antimicrobial agent and antioxidant.",
    "alcohol": "Percent of alcohol volume contained in the wine.",
}

PRESETS = {
    "Default (First Record)": {
        "fixed_acidity": 7.4, "volatile_acidity": 0.70, "citric_acid": 0.00,
        "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4
    },
    "High-Quality Reserve (Known Good Wine)": {
        "fixed_acidity": 5.2, "volatile_acidity": 0.34, "citric_acid": 0.00,
        "residual_sugar": 1.8, "chlorides": 0.050, "free_sulfur_dioxide": 27.0,
        "total_sulfur_dioxide": 63.0, "density": 0.9916, "pH": 3.68,
        "sulphates": 0.79, "alcohol": 14.0
    },
    "Dataset Average Wine": {
        "fixed_acidity": 7.22, "volatile_acidity": 0.34, "citric_acid": 0.32,
        "residual_sugar": 5.44, "chlorides": 0.056, "free_sulfur_dioxide": 30.5,
        "total_sulfur_dioxide": 115.7, "density": 0.9947, "pH": 3.22,
        "sulphates": 0.53, "alcohol": 10.49
    },
    "High Volatile Acidity (Flawed / Bad Wine)": {
        "fixed_acidity": 7.8, "volatile_acidity": 0.88, "citric_acid": 0.04,
        "residual_sugar": 2.3, "chlorides": 0.090, "free_sulfur_dioxide": 6.0,
        "total_sulfur_dioxide": 18.0, "density": 0.9980, "pH": 3.42,
        "sulphates": 0.48, "alcohol": 9.2
    }
}

# ---------------------------------------------------------
# Custom Styling:
# - Dark Wine Background with Crisp White Direct Text
# - Noticeably Whiter Frosted Glassmorphism Boxes
# - Guaranteed Text Visibility (No Invisible Text Anywhere)
# - Neon Glowing Green & Glowing Red Selection Callouts
# - Violet Buttons with White Text
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Wine Canvas Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #2f081c 0%, #1a040e 100%) !important;
        color: #ffffff !important;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    /* DIRECT BACKGROUND TEXT: Crisp White */
    .outer-title {
        color: #ffffff !important;
        font-size: 1.45rem;
        font-weight: 850;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.7);
    }
    .outer-subtitle {
        color: #fce8f0 !important;
        font-size: 0.98rem;
        margin-bottom: 1.3rem;
        font-weight: 500;
        text-shadow: 0 1px 6px rgba(0, 0, 0, 0.6);
    }

    /* =========================================================
       WHITER TRANSLUCENT BOXES (FROSTED GLASSMORPHISM)
       ========================================================= */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.28) 0%, rgba(255, 255, 255, 0.16) 100%) !important;
        backdrop-filter: blur(18px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(160%) !important;
        border: 2px solid rgba(255, 255, 255, 0.45) !important;
        border-radius: 20px !important;
        padding: 1.8rem 2rem !important;
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.45), 0 0 25px rgba(255, 255, 255, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        margin-bottom: 1.4rem !important;
        color: #ffffff !important;
    }
    .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4, .glass-card h5 {
        color: #ffffff !important;
        font-weight: 850;
        margin-top: 0;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
    }
    .glass-card p, .glass-card span, .glass-card label, .glass-card div {
        color: #ffffff !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6);
    }

    /* Universal Text Visibility Overrides */
    .stCaption, div[data-testid="stCaptionContainer"] p, div[data-testid="stCaptionContainer"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.7);
    }
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] span {
        color: #ffffff !important;
        font-weight: 750 !important;
        font-size: 0.95rem !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.75);
    }

    /* KPI / Metric Whiter Translucent Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.14) 100%) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 2px solid rgba(255, 255, 255, 0.38) !important;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.32), 0 0 15px rgba(255, 255, 255, 0.08);
        margin-bottom: 0.8rem;
    }
    .kpi-label {
        font-size: 0.86rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.7);
    }
    .kpi-value {
        font-size: 2.15rem;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
    }
    .kpi-sub {
        font-size: 0.88rem;
        font-weight: 700;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6);
    }

    /* Positive (Green) and Negative (Red) Text Highlighting */
    .text-positive, .glass-card .text-positive {
        color: #4ade80 !important; /* High-contrast bright green */
        font-weight: 850 !important;
        text-shadow: 0 0 10px rgba(34, 197, 94, 0.5), 0 1px 4px rgba(0, 0, 0, 0.8) !important;
    }
    .highlight-positive {
        background-color: rgba(20, 83, 45, 0.8) !important;
        color: #86efac !important;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 850;
        border: 1.5px solid #22c55e;
        display: inline-block;
        box-shadow: 0 0 14px rgba(34, 197, 94, 0.45);
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.7);
    }
    .kpi-border-positive {
        border-color: #22c55e !important;
        border-left: 6px solid #22c55e !important;
        box-shadow: 0 0 22px rgba(34, 197, 94, 0.35) !important;
    }

    .text-negative, .glass-card .text-negative {
        color: #f87171 !important; /* High-contrast bright red */
        font-weight: 850 !important;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.5), 0 1px 4px rgba(0, 0, 0, 0.8) !important;
    }
    .highlight-negative {
        background-color: rgba(127, 29, 29, 0.8) !important;
        color: #fca5a5 !important;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 850;
        border: 1.5px solid #ef4444;
        display: inline-block;
        box-shadow: 0 0 14px rgba(239, 68, 68, 0.45);
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.7);
    }
    .kpi-border-negative {
        border-color: #ef4444 !important;
        border-left: 6px solid #ef4444 !important;
        box-shadow: 0 0 22px rgba(239, 68, 68, 0.35) !important;
    }

    .text-neutral, .glass-card .text-neutral {
        color: #ffffff !important;
        font-weight: 850 !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.7);
    }

    /* =========================================================
       POPUP BOXES: NEON GLOWING GREEN & GLOWING RED (NOT WHITE)
       ========================================================= */
    .callout-valid-glow {
        background: linear-gradient(135deg, rgba(13, 51, 25, 0.94) 0%, rgba(20, 71, 35, 0.9) 100%) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 2px solid #22c55e !important;
        box-shadow: 0 0 28px rgba(34, 197, 94, 0.6), inset 0 0 14px rgba(34, 197, 94, 0.2) !important;
        padding: 1.15rem 1.4rem !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.96rem;
        margin-bottom: 0.5rem;
    }
    .callout-valid-glow b {
        color: #4ade80 !important;
        font-weight: 850;
    }
    .pill-green-glow {
        background: #166534;
        color: #86efac;
        border: 1.5px solid #22c55e;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 850;
        display: inline-block;
        box-shadow: 0 0 14px rgba(34, 197, 94, 0.55);
        letter-spacing: 0.04em;
    }

    .callout-error-glow {
        background: linear-gradient(135deg, rgba(61, 15, 23, 0.94) 0%, rgba(76, 19, 30, 0.9) 100%) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 2px solid #ef4444 !important;
        box-shadow: 0 0 28px rgba(239, 68, 68, 0.6), inset 0 0 14px rgba(239, 68, 68, 0.2) !important;
        padding: 1.15rem 1.4rem !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.96rem;
        margin-bottom: 0.5rem;
    }
    .callout-error-glow b {
        color: #f87171 !important;
        font-weight: 850;
    }
    .pill-red-glow {
        background: #991b1b;
        color: #fca5a5;
        border: 1.5px solid #ef4444;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 850;
        display: inline-block;
        box-shadow: 0 0 14px rgba(239, 68, 68, 0.55);
        letter-spacing: 0.04em;
    }

    /* Customized Prediction Verdict Glowing Banners */
    .status-banner-glowing {
        border-radius: 16px;
        padding: 1.45rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .status-banner-glowing.banner-positive {
        background: linear-gradient(135deg, rgba(13, 51, 25, 0.94) 0%, rgba(21, 74, 36, 0.9) 100%) !important;
        backdrop-filter: blur(14px) !important;
        border: 2.5px solid #22c55e !important;
        box-shadow: 0 0 28px rgba(34, 197, 94, 0.55) !important;
        color: #ffffff !important;
    }
    .status-banner-glowing.banner-negative {
        background: linear-gradient(135deg, rgba(61, 15, 23, 0.94) 0%, rgba(76, 19, 30, 0.9) 100%) !important;
        backdrop-filter: blur(14px) !important;
        border: 2.5px solid #ef4444 !important;
        box-shadow: 0 0 28px rgba(239, 68, 68, 0.55) !important;
        color: #ffffff !important;
    }

    /* =========================================================
       VIOLET BUTTONS WITH CRISP WHITE TEXT
       ========================================================= */
    div.stButton > button,
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="secondary"],
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
        color: #ffffff !important;
        border: 1.5px solid #8b5cf6 !important;
        font-weight: 750 !important;
        border-radius: 12px !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.45) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button p,
    div.stButton > button span,
    div[data-testid="stDownloadButton"] > button p,
    div[data-testid="stDownloadButton"] > button span {
        color: #ffffff !important;
        font-weight: 750 !important;
    }
    div.stButton > button:hover,
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #5b21b6 0%, #6d28d9 100%) !important;
        color: #ffffff !important;
        border-color: #a78bfa !important;
        box-shadow: 0 0 24px rgba(124, 58, 237, 0.75) !important;
        transform: translateY(-1px);
    }

    /* Inputs in Whiter Translucent Cards */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #c4b5fd !important;
        box-shadow: 0 0 18px rgba(124, 58, 237, 0.45) !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    div[data-testid="stNumberInput"] button svg {
        fill: #ffffff !important;
    }

    /* Expanders & Dataframe */
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.14) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stExpander"] details summary {
        color: #ffffff !important;
        font-weight: 750 !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6);
    }
    div[data-testid="stExpander"] details summary svg {
        fill: #ffffff !important;
    }
    div[data-testid="stDataFrame"] {
        background-color: rgba(0, 0, 0, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Model Training
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading wine dataset…")
def load_dataset():
    """Fetch UCI Wine Quality dataset ID 186, using local cache when present."""
    local_path = "wine_quality_dataset.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        wine = fetch_ucirepo(id=186)
        features = wine.data.features.copy()
        targets = wine.data.targets.iloc[:, 0].copy()
        features.columns = [str(c).strip() for c in features.columns]
        features = features[FEATURES].apply(pd.to_numeric, errors="raise")
        targets = pd.to_numeric(targets, errors="raise").astype(int)
        df = features.copy()
        df["quality"] = targets.to_numpy()
        df.to_csv(local_path, index=False)
    return df


@st.cache_resource(show_spinner="Training machine learning models…")
def train_models():
    """Train both Logistic Regression models (Binary Good/Bad & Multinomial Rating)."""
    df = load_dataset()
    X = df[FEATURES]
    y_rating = df["quality"].to_numpy()
    y_good_bad = (y_rating >= 7).astype(int)

    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_good_bad, y_rating, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    binary_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    binary_model.fit(X_train_scaled, y_bin_train)

    multinomial_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    multinomial_model.fit(X_train_scaled, y_multi_train)

    binary_accuracy = binary_model.score(X_test_scaled, y_bin_test)
    multi_accuracy = multinomial_model.score(X_test_scaled, y_multi_test)

    return df, scaler, binary_model, multinomial_model, binary_accuracy, multi_accuracy


def predict_frame(frame, scaler, binary_model, multinomial_model):
    """Run both models on a dataframe of wine features."""
    scaled = scaler.transform(frame[FEATURES])
    binary = binary_model.predict(scaled)
    ratings = multinomial_model.predict(scaled)
    probabilities = binary_model.predict_proba(scaled)
    good_class_idx = list(binary_model.classes_).index(1)

    result = frame[FEATURES].copy()
    result["Predicted Quality"] = np.where(binary == 1, "Good (Rating ≥ 7)", "Bad (Rating ≤ 6)")
    result["Predicted Quality Label"] = np.where(binary == 1, "Good", "Bad")
    result["Good Probability (%)"] = np.round(probabilities[:, good_class_idx] * 100, 1)
    result["Predicted Rating"] = ratings.astype(int)
    return result


# ---------------------------------------------------------
# Plotly Visualizations on Whiter Frosted Glass
# ---------------------------------------------------------
TRANSLUCENT_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.18)",
    font=dict(color="#ffffff", family="sans-serif", size=12),
)

def create_good_vs_bad_bar_chart(results):
    """Chart 1: Explicit comparison bar chart of Good vs Bad wine counts."""
    counts = results["Predicted Quality Label"].value_counts().reindex(["Good", "Bad"], fill_value=0).reset_index()
    counts.columns = ["Verdict", "Bottles"]
    total = len(results)
    counts["Percentage"] = (counts["Bottles"] / total * 100).round(1)
    counts["LabelText"] = counts.apply(lambda r: f"{r['Bottles']:,} ({r['Percentage']}%)", axis=1)

    fig = px.bar(
        counts,
        x="Verdict",
        y="Bottles",
        color="Verdict",
        color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
        text="LabelText",
    )
    fig.update_layout(
        **TRANSLUCENT_CHART_LAYOUT,
        height=340,
        margin=dict(l=15, r=15, t=25, b=15),
        showlegend=False,
        xaxis_title="Quality Classification Verdict",
        yaxis_title="Bottle Count",
        xaxis=dict(tickfont=dict(color="#ffffff", size=13, weight="bold"), gridcolor="rgba(255,255,255,0.14)"),
        yaxis=dict(tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.14)"),
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#ffffff", size=13, weight="bold"),
        marker_line_width=1.5,
        marker_line_color="rgba(255,255,255,0.5)",
    )
    return fig


def create_rating_distribution_chart(results):
    """Chart 2: Rating distribution showing BOTH Good (Green) & Bad (Red)."""
    counts = (
        results.groupby(["Predicted Rating", "Predicted Quality Label"])
        .size()
        .reset_index(name="Number of Bottles")
    )
    fig = px.bar(
        counts,
        x="Predicted Rating",
        y="Number of Bottles",
        color="Predicted Quality Label",
        color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
        text="Number of Bottles",
        barmode="stack",
    )
    fig.update_layout(
        **TRANSLUCENT_CHART_LAYOUT,
        height=340,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_title="Predicted Sensory Rating Score (Scale 3 - 9)",
        yaxis_title="Bottle Count",
        legend_title="Classification",
        xaxis=dict(tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.14)"),
        yaxis=dict(tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.14)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(color="#ffffff")),
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#ffffff", size=12, weight="bold"),
    )
    return fig


def create_attribute_comparison_chart(results):
    """Chart 3: Key chemical attributes comparison between Good (Green) and Bad (Red)."""
    features_to_compare = ["alcohol", "volatile_acidity", "citric_acid", "sulphates"]
    grouped = results.groupby("Predicted Quality Label")[features_to_compare].mean().reset_index()

    melted = pd.melt(
        grouped,
        id_vars=["Predicted Quality Label"],
        value_vars=features_to_compare,
        var_name="Feature",
        value_name="Average Value",
    )
    melted["Feature"] = melted["Feature"].map(DISPLAY_NAMES)

    fig = px.bar(
        melted,
        x="Feature",
        y="Average Value",
        color="Predicted Quality Label",
        barmode="group",
        color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"},
        text="Average Value",
    )
    fig.update_layout(
        **TRANSLUCENT_CHART_LAYOUT,
        height=340,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_title="",
        yaxis_title="Mean Chemical Level",
        legend_title="Quality Class",
        xaxis=dict(tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.14)"),
        yaxis=dict(tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.14)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#ffffff")),
    )
    fig.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside",
        textfont=dict(color="#ffffff", size=12, weight="bold")
    )
    return fig


def create_radar_chart(custom_values, df):
    """Chart 4: Radar benchmark comparison showing Custom Wine, Good Avg (Green), and Bad Avg (Red)."""
    good_mean = df[df["quality"] >= 7][FEATURES].mean()
    bad_mean = df[df["quality"] < 7][FEATURES].mean()
    min_vals = df[FEATURES].min()
    max_vals = df[FEATURES].max()

    def normalize(val_dict):
        norm = []
        for f in FEATURES:
            range_val = max_vals[f] - min_vals[f]
            if range_val == 0:
                norm.append(0.5)
            else:
                norm.append(float(np.clip((val_dict[f] - min_vals[f]) / range_val, 0.0, 1.0)))
        return norm

    custom_norm = normalize(custom_values)
    good_norm = normalize(good_mean.to_dict())
    bad_norm = normalize(bad_mean.to_dict())

    categories = [DISPLAY_NAMES[f] for f in FEATURES]
    categories_closed = categories + [categories[0]]
    custom_norm_closed = custom_norm + [custom_norm[0]]
    good_norm_closed = good_norm + [good_norm[0]]
    bad_norm_closed = bad_norm + [bad_norm[0]]

    fig = go.Figure()

    # Red for Bad Average (Negative benchmark)
    fig.add_trace(go.Scatterpolar(
        r=bad_norm_closed,
        theta=categories_closed,
        name="Dataset Bad Wine Avg (Negative)",
        line=dict(color="#ef4444", dash="dot", width=2.4),
        fill="none",
    ))
    # Green for Good Average (Positive benchmark)
    fig.add_trace(go.Scatterpolar(
        r=good_norm_closed,
        theta=categories_closed,
        name="Dataset Good Wine Avg (Positive)",
        line=dict(color="#22c55e", dash="dash", width=2.6),
        fill="none",
    ))
    # Violet for Custom Wine
    fig.add_trace(go.Scatterpolar(
        r=custom_norm_closed,
        theta=categories_closed,
        name="Your Custom Wine Sample",
        fill="toself",
        fillcolor="rgba(124, 58, 237, 0.4)",
        line=dict(color="#c4b5fd", width=3.2),
    ))

    fig.update_layout(
        **TRANSLUCENT_CHART_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0.22)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor="rgba(255,255,255,0.3)", gridcolor="rgba(255,255,255,0.16)"),
            angularaxis=dict(tickfont=dict(color="#ffffff", size=11, weight="bold"), linecolor="rgba(255,255,255,0.3)", gridcolor="rgba(255,255,255,0.16)"),
        ),
        showlegend=True,
        height=420,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(color="#ffffff")),
    )
    return fig


def create_custom_attribute_bars(custom_values):
    """Horizontal bar chart showing the entered attribute values."""
    chart_df = pd.DataFrame({
        "Attribute": [DISPLAY_NAMES[f] for f in FEATURES],
        "Value": [custom_values[f] for f in FEATURES],
        "Unit": [UNITS[f] for f in FEATURES],
    })
    fig = px.bar(
        chart_df,
        x="Value",
        y="Attribute",
        orientation="h",
        color="Value",
        color_continuous_scale=["#a78bfa", "#7c3aed"],
        text="Value",
    )
    fig.update_layout(
        **TRANSLUCENT_CHART_LAYOUT,
        height=420,
        margin=dict(l=15, r=15, t=25, b=15),
        coloraxis_showscale=False,
        yaxis=dict(
            categoryorder="array",
            categoryarray=[DISPLAY_NAMES[f] for f in FEATURES[::-1]],
            tickfont=dict(color="#ffffff", size=12, weight="bold"),
            gridcolor="rgba(255,255,255,0.14)",
        ),
        xaxis=dict(tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.14)"),
        xaxis_title="Configured Chemical Value",
        yaxis_title="",
    )
    fig.update_traces(
        texttemplate="%{x:.4g}",
        textposition="outside",
        textfont=dict(color="#ffffff", size=12, weight="bold")
    )
    return fig


def create_probability_gauge(prob_val):
    """Gauge chart for Good wine probability (Green if positive, Red if negative)."""
    color_val = "#22c55e" if prob_val >= 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_val,
        number={"suffix": "%", "font": {"size": 38, "color": color_val}},
        title={"text": "Estimated Good Quality Probability", "font": {"size": 16, "color": "#ffffff"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.4)", "tickfont": {"color": "#ffffff"}},
            "bar": {"color": "#a78bfa", "thickness": 0.38},
            "bgcolor": "rgba(0,0,0,0.2)",
            "borderwidth": 1.5,
            "bordercolor": "rgba(255,255,255,0.3)",
            "steps": [
                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.32)"},
                {"range": [50, 100], "color": "rgba(34, 197, 94, 0.32)"},
            ],
            "threshold": {
                "line": {"color": "#22c55e", "width": 4},
                "thickness": 0.75,
                "value": 50.0,
            },
        },
    ))
    fig.update_layout(
        **TRANSLUCENT_CHART_LAYOUT,
        height=260,
        margin=dict(l=20, r=20, t=35, b=15),
    )
    return fig


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "dataset_results" not in st.session_state:
    st.session_state["dataset_results"] = None
if "dataset_count" not in st.session_state:
    st.session_state["dataset_count"] = 0
if "show_custom" not in st.session_state:
    st.session_state["show_custom"] = False
if "custom_result" not in st.session_state:
    st.session_state["custom_result"] = None
if "custom_values" not in st.session_state:
    st.session_state["custom_values"] = None
if "preset_loaded" not in st.session_state:
    st.session_state["preset_loaded"] = "Default (First Record)"

# ---------------------------------------------------------
# Load Data & Train Models
# ---------------------------------------------------------
try:
    df, scaler, binary_model, multinomial_model, binary_accuracy, multi_accuracy = train_models()
except Exception as exc:
    st.error("Failed to load dataset or initialize models. Please verify your environment and data file.")
    st.exception(exc)
    st.stop()

total_bottles = len(df)

# ---------------------------------------------------------
# Hero Banner (Main Whiter Translucent Card)
# ---------------------------------------------------------
st.markdown("""
<div class="glass-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span style="background: rgba(124, 58, 237, 0.35); color: #ddd6fe; border: 1.5px solid #a78bfa; padding: 4px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;">
            Dual-Model Machine Learning · Enology AI
        </span>
        <span style="font-size: 0.92rem; font-weight: 700; color: #ffffff;">
            UCI Benchmark Dataset: <b style="color: #fef08a;">6,497 Records</b> · 11 Attributes
        </span>
    </div>
    <h1 style="color: #ffffff !important; font-size: 2.5rem; font-weight: 850; margin: 0.2rem 0 0.5rem 0;">
        🍷 Wine Quality Studio
    </h1>
    <p style="color: #ffffff !important; font-size: 1.05rem; margin-bottom: 0; line-height: 1.55;">
        Evaluate wine quality across 11 physicochemical attributes using dual Logistic Regression models:
        a <b style="color: #4ade80;">Binary Classifier</b> (Positive: Good ≥ 7 vs Negative: Bad ≤ 6) and a <b>Multinomial Classifier</b> (Sensory Rating 3 - 9).
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SECTION 1: DATASET DOWNLOAD & PREDICTION
# =========================================================
# Direct Background Header: Crisp White Text
st.markdown('<div class="outer-title">📥 1. Wine Dataset & Batch Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="outer-subtitle">Download the benchmark UCI Wine Quality dataset, specify your bottle sample size, and predict using both trained models.</div>',
    unsafe_allow_html=True
)

# Main Box for Dataset Controls in Whiter Translucent Glass
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col_data_left, col_data_right = st.columns([1.3, 1], gap="medium")

with col_data_left:
    st.markdown("#### Download Raw Dataset")
    st.write(
        f"The official dataset contains **{total_bottles:,} wine bottles** characterized by **11 chemical attributes** plus sensory quality ratings. "
        "Download the raw dataset directly as CSV."
    )
    csv_raw_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Wine Dataset (CSV)",
        data=csv_raw_data,
        file_name="wine_quality_uci_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_download_raw_dataset"
    )

with col_data_right:
    st.markdown("#### Dataset Overview")
    mo1, mo2 = st.columns(2)
    with mo1:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.18); border: 1.5px solid rgba(255, 255, 255, 0.35); border-radius: 12px; padding: 0.85rem; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 800; color: #fef08a; text-transform: uppercase;">Available Bottles</div>
            <div style="font-size: 1.65rem; font-weight: 850; color: #ffffff; text-shadow: 0 1px 6px rgba(0,0,0,0.6);">{total_bottles:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with mo2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.18); border: 1.5px solid rgba(255, 255, 255, 0.35); border-radius: 12px; padding: 0.85rem; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 800; color: #fef08a; text-transform: uppercase;">Attributes</div>
            <div style="font-size: 1.65rem; font-weight: 850; color: #ffffff; text-shadow: 0 1px 6px rgba(0,0,0,0.6);">11 Features</div>
        </div>
        """, unsafe_allow_html=True)
    st.caption("Source: UCI Machine Learning Repository · Dataset ID 186")

with st.expander("🔍 Preview First 5 Rows of Dataset"):
    st.dataframe(df.head(5), use_container_width=True)

st.markdown("<hr style='margin: 1.3rem 0; border-color: rgba(255,255,255,0.25);'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Bottle Count Input & Glowing Selection Box (Glowing Green)
# ---------------------------------------------------------
st.markdown("#### Enter Number of Bottles to Predict")
st.write(
    f"Specify the number of bottles to test (1 to {total_bottles:,}). "
    "The application calculates and verifies your selection to ensure it does not exceed the total available bottles."
)

c_input, c_status = st.columns([1, 1.25], gap="large")

with c_input:
    bottle_count_input = st.number_input(
        "Number of bottles to analyze:",
        min_value=1,
        max_value=total_bottles + 10000,
        value=min(50, total_bottles),
        step=10,
        help=f"Dataset contains {total_bottles:,} records in total.",
        key="input_bottle_count"
    )

bottle_count_int = int(bottle_count_input)
is_valid_count = (1 <= bottle_count_int <= total_bottles)
count_exceeded = (bottle_count_int > total_bottles)

with c_status:
    st.write("**Selection Status & Calculations:**")
    if count_exceeded:
        # Glowing Red Error Box (Neon Red, Not White)
        st.markdown(
            f"""<div class="callout-error-glow">
                <span class="pill-red-glow">⚠️ LIMIT EXCEEDED (Negative)</span><br>
                <div style="margin-top: 0.5rem; line-height: 1.5; color: #ffffff;">
                    You requested <b style="color: #fca5a5;">{bottle_count_int:,}</b> bottles, but the dataset only has <b>{total_bottles:,}</b> bottles.<br>
                    <span style="color: #fecaca;">Please enter a number between 1 and {total_bottles:,}.</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )
    elif bottle_count_int < 1:
        st.markdown(
            """<div class="callout-error-glow">
                <span class="pill-red-glow">⚠️ INVALID COUNT</span>: Please enter at least 1 bottle.
            </div>""",
            unsafe_allow_html=True
        )
    else:
        # Glowing Green Valid Box (Neon Green, Not White)
        percentage = (bottle_count_int / total_bottles) * 100
        remaining = total_bottles - bottle_count_int
        st.markdown(
            f"""<div class="callout-valid-glow">
                <span class="pill-green-glow">✅ VALID SELECTION (Positive)</span><br>
                <div style="margin-top: 0.5rem; line-height: 1.5; color: #ffffff;">
                    Ready to predict <b style="color: #4ade80; font-size: 1.05rem;">{bottle_count_int:,}</b> bottles (<b style="color: #4ade80;">{percentage:.2f}%</b> of dataset).<br>
                    <span style="color: #dcfce7; font-size: 0.9rem;">{remaining:,} remaining bottles will be unselected.</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

# Predict Button (Violet with White Text)
btn_col1, btn_col2 = st.columns([1.5, 1])
with btn_col1:
    predict_dataset_clicked = st.button(
        f"🚀 Predict {bottle_count_int:,} Dataset Bottles with Both Models",
        type="primary",
        disabled=not is_valid_count,
        use_container_width=True,
        key="btn_predict_dataset"
    )

with btn_col2:
    if st.session_state["dataset_results"] is not None:
        if st.button("🔄 Clear Prediction Results", use_container_width=True, key="btn_clear_dataset"):
            st.session_state["dataset_results"] = None
            st.session_state["dataset_count"] = 0
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True) # End of Dataset Input Whiter Glass Card

if predict_dataset_clicked and is_valid_count:
    selected_sample = df.iloc[:bottle_count_int][FEATURES].copy()
    st.session_state["dataset_results"] = predict_frame(
        selected_sample, scaler, binary_model, multinomial_model
    )
    st.session_state["dataset_count"] = bottle_count_int
    st.toast(f"Successfully predicted {bottle_count_int:,} bottles!", icon="🍷")

# ---------------------------------------------------------
# Dataset Prediction Dashboard (Whiter Translucent Boxes with Green/Red Highlighting)
# ---------------------------------------------------------
if st.session_state["dataset_results"] is not None:
    results_df = st.session_state["dataset_results"]
    count_predicted = st.session_state["dataset_count"]

    # Direct Background Header: Crisp White Text
    st.markdown('<div class="outer-title">📊 Dataset Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="outer-subtitle">Summary metrics and individual predictions for the selected <b>{count_predicted:,} bottles</b>:</div>', unsafe_allow_html=True)

    good_count = int((results_df["Predicted Quality Label"] == "Good").sum())
    bad_count = count_predicted - good_count
    good_pct = (good_count / count_predicted * 100) if count_predicted > 0 else 0
    bad_pct = (bad_count / count_predicted * 100) if count_predicted > 0 else 0
    avg_rating = float(results_df["Predicted Rating"].mean())

    # 4 Metric Cards in Whiter Translucent Boxes with Green/Red Highlighting
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Bottles Evaluated</div>
            <div class="kpi-value text-neutral">{count_predicted:,}</div>
            <div class="kpi-sub text-neutral">Selected Sample</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        # Green highlight for Positive (Good wines)
        st.markdown(f"""
        <div class="kpi-card kpi-border-positive">
            <div class="kpi-label text-positive">Predicted Good (≥7)</div>
            <div class="kpi-value text-positive">{good_count:,}</div>
            <div class="kpi-sub"><span class="highlight-positive">▲ {good_pct:.1f}% Positive</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        # Red highlight for Negative (Bad wines)
        st.markdown(f"""
        <div class="kpi-card kpi-border-negative">
            <div class="kpi-label text-negative">Predicted Bad (≤6)</div>
            <div class="kpi-value text-negative">{bad_count:,}</div>
            <div class="kpi-sub"><span class="highlight-negative">▼ {bad_pct:.1f}% Negative</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        avg_class = "text-positive" if avg_rating >= 6.5 else ("text-negative" if avg_rating < 5.5 else "text-neutral")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Rating</div>
            <div class="kpi-value {avg_class}">{avg_rating:.2f} <span style="font-size: 1.15rem; color: #fef08a;">/ 9</span></div>
            <div class="kpi-sub text-neutral">Multinomial Model</div>
        </div>
        """, unsafe_allow_html=True)

    # Whiter Translucent Box for Table & CSV Download
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size: 1rem; font-weight: 700; color: #ffffff; text-shadow: 0 1px 4px rgba(0,0,0,0.6);'>"
        f"🎯 Model Test Split Accuracies: Binary Classifier: <b class='text-positive'>{binary_accuracy:.2%}</b> · "
        f"Exact Rating Classifier: <b style='color: #fef08a;'>{multi_accuracy:.2%}</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    with st.expander(f"📋 View Bottle-by-Bottle Predictions Table ({count_predicted:,} Rows)"):
        st.dataframe(results_df, use_container_width=True, height=320)

    csv_predictions = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download Prediction Results for {count_predicted:,} Bottles (CSV)",
        data=csv_predictions,
        file_name=f"wine_predictions_{count_predicted}_bottles.csv",
        mime="text/csv",
        use_container_width=True,
        key="btn_download_dataset_preds"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SEPARATE GRAPH SECTION (SHOWING BOTH GOOD AND BAD)
    # ---------------------------------------------------------
    st.markdown('<div class="outer-title">📈 Visual Analytics: Prediction Graphs (Good vs Bad)</div>', unsafe_allow_html=True)
    st.markdown('<div class="outer-subtitle">Visual analytics displaying both Good and Bad prediction breakdowns across your selection.</div>', unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2, gap="medium")
    with g_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 1. Good (Green) vs. Bad (Red) Bottle Counts")
        st.caption("Side-by-side count comparing bottles classified as Good versus Bad.")
        st.plotly_chart(create_good_vs_bad_bar_chart(results_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 2. Predicted Rating Distribution (Good & Bad)")
        st.caption("Distribution of predicted ratings colored by Good (Green) and Bad (Red).")
        st.plotly_chart(create_rating_distribution_chart(results_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 3. Key Chemical Attribute Comparison (Good: Green vs. Bad: Red)")
    st.caption("Mean alcohol, volatile acidity, citric acid, and sulphates for Good vs Bad predictions.")
    st.plotly_chart(create_attribute_comparison_chart(results_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()


# =========================================================
# SECTION 2: PREDICT CUSTOMIZED DATA (11 ATTRIBUTES)
# =========================================================
# Direct Background Header: Crisp White Text
st.markdown('<div class="outer-title">🧪 2. Predict Customized Wine Data</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="outer-subtitle">Customize any or all of the 11 chemical attributes at any time to test bespoke wine samples using both trained models.</div>',
    unsafe_allow_html=True
)

# Open / Close Toggle Button (Violet with White Text)
col_tgl1, col_tgl2 = st.columns([1.5, 1])
with col_tgl1:
    btn_label = "❌ Close Customized Prediction Panel" if st.session_state["show_custom"] else "✨ Open Customized Wine Predictor"
    if st.button(btn_label, use_container_width=True, key="btn_toggle_custom"):
        st.session_state["show_custom"] = not st.session_state["show_custom"]
        st.rerun()

with col_tgl2:
    if st.session_state["show_custom"] and st.session_state["custom_result"] is not None:
        if st.button("🔄 Clear Customized Prediction", use_container_width=True, key="btn_clear_custom"):
            st.session_state["custom_result"] = None
            st.session_state["custom_values"] = None
            st.rerun()

if st.session_state["show_custom"]:
    # Main Box for 11 Chemical Attributes Configurator in Whiter Translucent Glass
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Select / Adjust the 11 Chemical Attributes")
    st.write(
        "You have full access to change any of the 11 chemical attributes below. "
        "Use the quick preset buttons to load known wine samples or modify individual parameters."
    )

    # Preset selection
    st.write("**Quick Presets:**")
    preset_cols = st.columns(4)
    for idx, (p_name, p_vals) in enumerate(PRESETS.items()):
        with preset_cols[idx]:
            if st.button(f"📌 {p_name}", use_container_width=True, key=f"btn_preset_{idx}"):
                for k, v in p_vals.items():
                    st.session_state[f"custom_{k}"] = float(v)
                st.session_state["preset_loaded"] = p_name
                st.rerun()

    st.caption(f"Currently loaded preset: **{st.session_state.get('preset_loaded', 'Default')}**")
    st.markdown("<hr style='margin: 0.9rem 0; border-color: rgba(255,255,255,0.25);'>", unsafe_allow_html=True)

    # Compute sensible min and max limits from dataset for each attribute
    ranges = {}
    for feature in FEATURES:
        lo = float(df[feature].min())
        hi = float(df[feature].max())
        pad = (hi - lo) * 0.10 if hi > lo else 1.0
        ranges[feature] = (max(0.0, lo - pad), hi + pad)

    # 11 Attributes organized in 3 clean columns
    custom_inputs = {}
    default_vals = PRESETS["Default (First Record)"]

    for row_start in range(0, len(FEATURES), 3):
        col_group = st.columns(3, gap="medium")
        for col, feature in zip(col_group, FEATURES[row_start:row_start + 3]):
            low_val, high_val = ranges[feature]
            def_val = float(default_vals[feature])
            key_name = f"custom_{feature}"

            if key_name not in st.session_state:
                st.session_state[key_name] = def_val

            fmt = "%.5f" if feature in ("density", "chlorides") else "%.4f"
            step_size = 0.0001 if feature in ("density", "chlorides") else 0.01

            with col:
                custom_inputs[feature] = st.number_input(
                    label=f"{DISPLAY_NAMES[feature]} ({UNITS[feature]})",
                    min_value=float(low_val),
                    max_value=float(high_val),
                    value=float(st.session_state[key_name]),
                    step=step_size,
                    format=fmt,
                    help=HELP_TEXT[feature],
                    key=key_name,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_custom_clicked = st.button(
        "🔮 Predict Customized Wine Quality with Both Models",
        type="primary",
        use_container_width=True,
        key="btn_run_custom_prediction"
    )

    if predict_custom_clicked:
        sample_df = pd.DataFrame([{f: custom_inputs[f] for f in FEATURES}], columns=FEATURES)
        custom_pred_result = predict_frame(sample_df, scaler, binary_model, multinomial_model)
        st.session_state["custom_result"] = custom_pred_result
        st.session_state["custom_values"] = custom_inputs
        st.toast("Customized prediction completed!", icon="✨")

    st.markdown('</div>', unsafe_allow_html=True) # End of Configurator Whiter Glass Card

    # ---------------------------------------------------------
    # Customized Prediction Dashboard (Glowing Banners & Whiter Translucent Cards)
    # ---------------------------------------------------------
    if st.session_state["custom_result"] is not None:
        c_res = st.session_state["custom_result"].iloc[0]
        quality_label = c_res["Predicted Quality Label"]
        rating_score = int(c_res["Predicted Rating"])
        good_prob = float(c_res["Good Probability (%)"])
        is_good = (quality_label == "Good")

        # Direct Background Header: Crisp White Text
        st.markdown('<div class="outer-title">📊 Customized Wine Prediction Dashboard</div>', unsafe_allow_html=True)

        # Glowing Neon Banner: Glowing Green if Positive, Glowing Red if Negative (Not White)
        if is_good:
            st.markdown(
                f"""<div class="status-banner-glowing banner-positive">
                    <div style="font-size: 1.55rem; font-weight: 850; margin-bottom: 0.4rem;">
                        🍷 <span class="pill-green-glow">PREDICTION: GOOD QUALITY WINE (Positive)</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                        Sensory Rating: <b style="color: #4ade80; font-size: 1.35rem;">{rating_score} / 9</b> (Positive Verdict) · Model Confidence: <b style="color: #4ade80;">{good_prob:.1f}% Good</b>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class="status-banner-glowing banner-negative">
                    <div style="font-size: 1.55rem; font-weight: 850; margin-bottom: 0.4rem;">
                        ⚠️ <span class="pill-red-glow">PREDICTION: BAD / STANDARD QUALITY WINE (Negative)</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                        Sensory Rating: <b style="color: #f87171; font-size: 1.35rem;">{rating_score} / 9</b> (Negative Verdict) · Model Confidence: <b style="color: #f87171;">{100 - good_prob:.1f}% Bad ({good_prob:.1f}% Good)</b>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )

        # 3 KPI Cards in Whiter Translucent Boxes with Green/Red Highlighting
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            if is_good:
                st.markdown("""
                <div class="kpi-card kpi-border-positive">
                    <div class="kpi-label text-positive">Binary Classification</div>
                    <div class="kpi-value text-positive">Good Quality</div>
                    <div class="kpi-sub"><span class="highlight-positive">Rating Threshold ≥ 7 (Positive)</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="kpi-card kpi-border-negative">
                    <div class="kpi-label text-negative">Binary Classification</div>
                    <div class="kpi-value text-negative">Bad Quality</div>
                    <div class="kpi-sub"><span class="highlight-negative">Rating Threshold ≤ 6 (Negative)</span></div>
                </div>
                """, unsafe_allow_html=True)

        with cd2:
            rating_class = "text-positive" if rating_score >= 7 else "text-negative"
            border_cls = "kpi-border-positive" if rating_score >= 7 else "kpi-border-negative"
            tag_cls = "highlight-positive" if rating_score >= 7 else "highlight-negative"
            verdict_text = "Positive Score" if rating_score >= 7 else "Negative Score"
            st.markdown(f"""
            <div class="kpi-card {border_cls}">
                <div class="kpi-label {rating_class}">Multinomial Quality Rating</div>
                <div class="kpi-value {rating_class}">{rating_score} <span style="font-size: 1.15rem; color: #fef08a;">/ 9</span></div>
                <div class="kpi-sub"><span class="{tag_cls}">{verdict_text}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with cd3:
            prob_class = "text-positive" if good_prob >= 50 else "text-negative"
            border_cls = "kpi-border-positive" if good_prob >= 50 else "kpi-border-negative"
            tag_cls = "highlight-positive" if good_prob >= 50 else "highlight-negative"
            st.markdown(f"""
            <div class="kpi-card {border_cls}">
                <div class="kpi-label {prob_class}">Good Wine Probability</div>
                <div class="kpi-value {prob_class}">{good_prob:.1f}%</div>
                <div class="kpi-sub"><span class="{tag_cls}">Model Confidence</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Whiter Translucent Card for Table & CSV Download
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.expander("📋 View Complete Feature & Prediction Breakdown"):
            st.dataframe(st.session_state["custom_result"], use_container_width=True)

        csv_custom = st.session_state["custom_result"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Customized Wine Prediction (CSV)",
            data=csv_custom,
            file_name="custom_wine_sample_prediction.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_custom_result"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------------
        # SEPARATE GRAPH SECTION FOR CUSTOMIZED PREDICTION (SHOWING BOTH GOOD & BAD)
        # ---------------------------------------------------------
        st.markdown('<div class="outer-title">📊 Customized Wine Visual Analytics & Benchmark Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="outer-subtitle">Comparison contrasting your customized wine against both Good and Bad benchmark standards.</div>', unsafe_allow_html=True)

        cg1, cg2 = st.columns([1.2, 1], gap="medium")
        with cg1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 1. Chemical Profile Radar: Custom vs. Good & Bad Benchmarks")
            st.caption("Normalized comparison: Green = Good Wine Average (Positive), Red = Bad Wine Average (Negative), Violet = Your Custom Sample.")
            st.plotly_chart(create_radar_chart(st.session_state["custom_values"], df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cg2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 2. Probability Gauge (Green: Positive, Red: Negative)")
            st.caption("Visual gauge indicating estimated probability of positive Good quality.")
            st.plotly_chart(create_probability_gauge(good_prob), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 3. Configured 11 Chemical Attribute Levels")
        st.plotly_chart(create_custom_attribute_bars(st.session_state["custom_values"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown(
    '<p style="color: #ffffff; text-align: center; font-size: 0.92rem; font-weight: 600; text-shadow: 0 1px 4px rgba(0,0,0,0.6);">'
    'Wine Quality Studio · Powered by scikit-learn & Streamlit · Binary & Multinomial Logistic Regression · UCI ID 186'
    '</p>',
    unsafe_allow_html=True
)
