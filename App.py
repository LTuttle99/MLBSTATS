import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# ---- Data source (FanGraphs via pybaseball) ----
from pybaseball import batting_stats, pitching_stats


# -----------------------------
# Utilities
# -----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan

def time_split(df, season_col="Season", train_end=2021, val_season=2022, test_season=2023):
    train = df[df[season_col] <= train_end].copy()
    val = df[df[season_col] == val_season].copy()
    test = df[df[season_col] == test_season].copy()
    return train, val, test

def build_model(model_name: str, numeric_features: list):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features)],
        remainder="drop"
    )

    if model_name == "Ridge":
        reg = Ridge(alpha=10.0, random_state=0)
    elif model_name == "Random Forest":
        reg = RandomForestRegressor(
            n_estimators=400,
            random_state=0,
            n_jobs=-1,
            min_samples_leaf=2
        )
    elif model_name == "Gradient Boosting":
        reg = GradientBoostingRegressor(random_state=0)
    else:
        reg = Ridge(alpha=10.0, random_state=0)

    return Pipeline([("pre", pre), ("model", reg)])

def add_lag_features(df, group_col, season_col, feature_cols, lags=(1, 2), rolling_windows=(2, 3)):
    """
    For each feature: add lag1/lag2 and rolling mean over past windows (excluding current season).
    """
    df = df.sort_values([group_col, season_col]).copy()

    for lag in lags:
        for c in feature_cols:
            df[f"{c}_lag{lag}"] = df.groupby(group_col)[c].shift(lag)

    for w in rolling_windows:
        for c in feature_cols:
            # rolling mean of prior seasons, so shift then rolling
            df[f"{c}_roll{w}"] = (
                df.groupby(group_col)[c]
                  .shift(1)
                  .rolling(window=w, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
    return df

def make_next_season_target(df, group_col, season_col, target_col, new_target_name="target_next"):
    df = df.sort_values([group_col, season_col]).copy()
    df[new_target_name] = df.groupby(group_col)[target_col].shift(-1)
    return df

def clean_columns_for_model(df):
    # Keep numeric-only feature columns for simplicity.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="MLB Player Predictive Models", layout="wide")

st.title("⚾ MLB Player Predictive Models (Next-Season Forecasts)")
st.caption("Pulls FanGraphs player-season data (via pybaseball), engineers lag/rolling features, and trains a regression model to predict next-season performance.")

with st.sidebar:
    st.header("Settings")

    player_type = st.selectbox("Player type", ["Hitters", "Pitchers"])
    start_season = st.number_input("Start season", min_value=2006, max_value=2025, value=2015, step=1)
    end_season = st.number_input("End season", min_value=2006, max_value=2025, value=2024, step=1)

    min_pa_ip = st.number_input(
        "Min playing time filter",
        min_value=0,
        max_value=800,
        value=200 if player_type == "Hitters" else 40,
        step=10,
        help="Hitters: PA threshold; Pitchers: IP threshold"
    )

    model_name = st.selectbox("Model", ["Ridge", "Random Forest", "Gradient Boosting"])

    st.subheader("Time Split")
    train_end = st.number_input("Train end season", min_value=int(start_season), max_value=int(end_season), value=min(2021, int(end_season)-2), step=1)
    val_season = st.number_input("Validation season", min_value=int(start_season), max_value=int(end_season), value=min(2022, int(end_season)-1), step=1)
    test_season = st.number_input("Test season", min_value=int(start_season), max_value=int(end_season), value=min(2023, int(end_season)), step=1)

    st.subheader("Feature engineering")
    use_lags = st.checkbox("Add lag features", value=True)
    use_rolls = st.checkbox("Add rolling averages", value=True)

    run_btn = st.button("Run Model", type="primary")


@st.cache_data(show_spinner=True)
def load_fg_data(player_type, start_season, end_season):
    if player_type == "Hitters":
        df = batting_stats(int(start_season), int(end_season))
    else:
        df = pitching_stats(int(start_season), int(end_season))
    return df


def pick_default_target_and_base_features(df, player_type):
    """
    Choose a reasonable target + feature set using columns commonly present in FanGraphs tables.
    If a column isn't present, we simply won't use it.
    """
    if player_type == "Hitters":
        # Preferred targets
        target_candidates = ["wOBA", "OPS", "wRC+", "HR"]
        base_feature_candidates = [
            "Age", "G", "PA", "HR", "R", "RBI", "SB",
            "BB%", "K%", "ISO", "BABIP", "AVG", "OBP", "SLG",
            "LD%", "GB%", "FB%", "Hard%", "Soft%", "Med%",
            "Pull%", "Cent%", "Oppo%"
        ]
        group_col = "IDfg"
        season_col = "Season"
        pt_col = "PA"
    else:
        target_candidates = ["FIP", "ERA", "K%", "BB%"]
        base_feature_candidates = [
            "Age", "G", "GS", "IP", "K/9", "BB/9", "HR/9",
            "K%", "BB%", "GB%", "FB%", "LD%", "Hard%", "Soft%", "Med%",
            "BABIP", "LOB%", "HR/FB"
        ]
        group_col = "IDfg"
        season_col = "Season"
        pt_col = "IP"

    # Choose first target that exists
    target = next((c for c in target_candidates if c in df.columns), None)
    if target is None:
        # fallback: pick any numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = numeric_cols[0] if numeric_cols else None

    # Build base features from candidates that exist
    base_features = [c for c in base_feature_candidates if c in df.columns]

    # Ensure group/season exist
    if group_col not in df.columns:
        # try alternative identifiers
        alt = [c for c in ["playerid", "playeridfg", "IDfg", "MLBID"] if c in df.columns]
        group_col = alt[0] if alt else None

    if season_col not in df.columns:
        alt = [c for c in ["Season", "year"] if c in df.columns]
        season_col = alt[0] if alt else None

    return target, base_features, group_col, season_col, pt_col


if run_btn:
    if end_season < start_season:
        st.error("End season must be >= start season.")
        st.stop()

    with st.spinner("Loading FanGraphs data..."):
        raw = load_fg_data(player_type, start_season, end_season)

    if raw is None or len(raw) == 0:
        st.error("No data returned. Try a different season range.")
        st.stop()

    target_default, base_features_default, group_col, season_col, pt_col = pick_default_target_and_base_features(raw, player_type)

    if group_col is None or season_col is None:
        st.error("Could not find player identifier / season columns in the pulled dataset.")
        st.stop()

    # Let user choose target & feature columns based on what exists
    with st.sidebar:
        st.subheader("Target & features")
        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox("Target stat (predict next season)", options=[c for c in numeric_cols if c not in [season_col]], index=(numeric_cols.index(target_default) if target_default in numeric_cols else 0))
        # Suggest features (numeric only, exclude target and season)
        suggested_features = [c for c in base_features_default if c in numeric_cols and c not in [target_col, season_col]]
        feature_cols = st.multiselect(
            "Base feature columns (current-season stats)",
            options=[c for c in numeric_cols if c not in [target_col]],
            default=suggested_features
        )

    # Filter for playing time and keep essential columns
    df = raw.copy()
    df = df.dropna(subset=[group_col, season_col])
    df[season_col] = df[season_col].astype(int)

    if pt_col in df.columns:
        df = df[df[pt_col] >= min_pa_ip].copy()

    # Keep columns we need
    keep_cols = list({group_col, "Name", season_col, target_col, *feature_cols})
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Make target = next season value of target_col
    df = make_next_season_target(df, group_col, season_col, target_col, new_target_name="target_next")

    # Add lag/rolling features built from base feature cols (and optionally the target col itself as a feature)
    base_for_fe = list(feature_cols)
    if target_col not in base_for_fe:
        base_for_fe.append(target_col)

    lags = (1, 2) if use_lags else tuple()
    rolls = (2, 3) if use_rolls else tuple()

    if use_lags or use_rolls:
        df = add_lag_features(
            df,
            group_col=group_col,
            season_col=season_col,
            feature_cols=base_for_fe,
            lags=lags if use_lags else ( ),
            rolling_windows=rolls if use_rolls else ( )
        )

    # Drop rows where next-season target is missing (last year in range)
    df_model = df.dropna(subset=["target_next"]).copy()

    # Also require at least some lag info if user enabled lags/rolls (avoid too many NAs)
    # We'll let imputer handle this, but dropping rows with all-null features helps.
    # Build numeric features list (exclude identifiers and the target_next itself)
    numeric_cols_model = df_model.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_cols_model if c not in ["target_next", season_col]]

    # Split
    train, val, test = time_split(df_model, season_col=season_col, train_end=int(train_end), val_season=int(val_season), test_season=int(test_season))

    if len(train) < 50 or len(val) < 20 or len(test) < 20:
        st.warning(
            f"Split sizes look small: train={len(train)}, val={len(val)}, test={len(test)}. "
            "Try widening the season range or lowering the playing-time filter."
        )

    X_train, y_train = train[numeric_features], train["target_next"]
    X_val, y_val = val[numeric_features], val["target_next"]
    X_test, y_test = test[numeric_features], test["target_next"]

    model = build_model(model_name, numeric_features)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val) if len(val) else np.array([])
    test_pred = model.predict(X_test) if len(test) else np.array([])

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Validation Metrics")
        if len(val):
            st.write({
                "RMSE": rmse(y_val, val_pred),
                "MAE": float(mean_absolute_error(y_val, val_pred)),
                "R2": safe_r2(y_val, val_pred),
                "n": int(len(val))
            })
        else:
            st.write("No validation rows in split.")
    with col2:
        st.subheader("Test Metrics")
        if len(test):
            st.write({
                "RMSE": rmse(y_test, test_pred),
                "MAE": float(mean_absolute_error(y_test, test_pred)),
                "R2": safe_r2(y_test, test_pred),
                "n": int(len(test))
            })
        else:
            st.write("No test rows in split.")
    with col3:
        st.subheader("Baseline (Persistence)")
        # baseline predicts next season = this season's target_col
        if len(test) and target_col in test.columns:
            baseline_pred = test[target_col].values
            st.write({
                "RMSE": rmse(y_test, baseline_pred),
                "MAE": float(mean_absolute_error(y_test, baseline_pred)),
                "R2": safe_r2(y_test, baseline_pred),
                "n": int(len(test))
            })
        else:
            st.write("Baseline unavailable for this split.")

    st.divider()

    # Predictions table
    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader(f"Top Predictions (Test Season = {int(test_season)})")
        if len(test):
            out = test[[group_col, season_col]].copy()
            if "Name" in test.columns:
                out["Name"] = test["Name"].values
            out["Predicted"] = test_pred
            out["Actual"] = y_test.values
            out["Error"] = out["Predicted"] - out["Actual"]
            out = out.sort_values("Predicted", ascending=False)
            st.dataframe(out.head(25), use_container_width=True)
        else:
            st.write("No test data to display.")

    with right:
        st.subheader("Pred vs Actual (Test)")
        if len(test):
            fig = plt.figure()
            plt.scatter(y_test, test_pred)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{model_name}: Actual vs Predicted")
            st.pyplot(fig)

            # If Ridge, show top coefficients
            if model_name == "Ridge":
                try:
                    ridge = model.named_steps["model"]
                    # Need to map coefficients back to original feature names (numeric_features)
                    coefs = pd.Series(ridge.coef_, index=numeric_features).sort_values(key=np.abs, ascending=False)
                    st.subheader("Top Ridge Coefficients (abs)")
                    st.dataframe(coefs.head(20).to_frame("coef"), use_container_width=True)
                except Exception:
                    pass
        else:
            st.write("No test data to plot.")

    st.divider()

    st.subheader("Dataset Preview")
    st.write("Modeling rows (after next-season target shift):", len(df_model))
    st.dataframe(df_model.head(20), use_container_width=True)
else:
    st.info("Set your options in the sidebar, then click **Run Model**.")
    st.markdown(
        """
**What this app does**
- Pulls FanGraphs player-season data using `pybaseball`
- Builds a next-season target (e.g., next year's wOBA / FIP)
- Engineers lag + rolling features
- Trains a regression model and evaluates on a time split
"""
    )
