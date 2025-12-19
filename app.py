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
from pybaseball import batting_stats, pitching_stats


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def time_split(df, season_col, train_end, val_season, test_season):
    train = df[df[season_col] <= train_end]
    val = df[df[season_col] == val_season]
    test = df[df[season_col] == test_season]
    return train, val, test

def build_model(model_name, numeric_features):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features)],
        remainder="drop"
    )

    if model_name == "Ridge":
        model = Ridge(alpha=10.0)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(random_state=0)

    return Pipeline([("pre", pre), ("model", model)])

st.set_page_config(page_title="MLB Predictive Models", layout="wide")
st.title("⚾ MLB Player Next-Season Prediction App")

with st.sidebar:
    player_type = st.selectbox("Player Type", ["Hitters", "Pitchers"])
    start_season = st.number_input("Start Season", 2006, 2025, 2015)
    end_season = st.number_input("End Season", 2006, 2025, 2024)
    model_name = st.selectbox("Model", ["Ridge", "Random Forest", "Gradient Boosting"])
    run = st.button("Run Model")

@st.cache_data
def load_data(player_type, start, end):
    if player_type == "Hitters":
        return batting_stats(start, end)
    else:
        return pitching_stats(start, end)

if run:
    df = load_data(player_type, start_season, end_season)
    df = df.dropna()

    target_col = "wOBA" if player_type == "Hitters" else "FIP"
    df["target_next"] = df.groupby("IDfg")[target_col].shift(-1)
    df = df.dropna(subset=["target_next"])

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [c for c in numeric_cols if c not in ["target_next", "Season"]]

    train, val, test = time_split(df, "Season", end_season-2, end_season-1, end_season)

    X_train, y_train = train[numeric_features], train["target_next"]
    X_test, y_test = test[numeric_features], test["target_next"]

    model = build_model(model_name, numeric_features)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.subheader("Test Performance")
    st.write({
        "RMSE": rmse(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    })

    out = test[["Name", "Season"]].copy()
    out["Predicted"] = preds
    out["Actual"] = y_test.values
    st.subheader("Top Predictions")
    st.dataframe(out.sort_values("Predicted", ascending=False).head(20))

    fig = plt.figure()
    plt.scatter(y_test, preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    st.pyplot(fig)
