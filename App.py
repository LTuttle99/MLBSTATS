import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gradio as gr

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pybaseball import batting_stats, pitching_stats


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def build_model(model_name: str, numeric_features):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([("num", num_pipe, numeric_features)], remainder="drop")

    if model_name == "Ridge":
        reg = Ridge(alpha=10.0, random_state=0)
    elif model_name == "Random Forest":
        reg = RandomForestRegressor(
            n_estimators=400,
            random_state=0,
            n_jobs=-1,
            min_samples_leaf=2
        )
    else:
        reg = GradientBoostingRegressor(random_state=0)

    return Pipeline([("pre", pre), ("model", reg)])


def load_fg(player_type: str, start_season: int, end_season: int) -> pd.DataFrame:
    if player_type == "Hitters":
        return batting_stats(start_season, end_season)
    return pitching_stats(start_season, end_season)


def pick_default_target(player_type: str, df: pd.DataFrame) -> str:
    if player_type == "Hitters":
        for c in ["wOBA", "wRC+", "OPS", "HR"]:
            if c in df.columns:
                return c
    else:
        for c in ["FIP", "ERA", "K%", "BB%"]:
            if c in df.columns:
                return c
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return num[0] if num else ""


def train_and_predict(
    player_type: str,
    start_season: int,
    end_season: int,
    min_playing_time: int,
    target_col: str,
    model_name: str,
    train_end: int,
    val_season: int,
    test_season: int,
    top_n: int
):
    df = load_fg(player_type, int(start_season), int(end_season))
    if df is None or len(df) == 0:
        raise gr.Error("No data returned from pybaseball. Try a different season range.")

    if "Season" not in df.columns:
        raise gr.Error("Expected column 'Season' not found in FanGraphs table returned.")
    if "IDfg" not in df.columns:
        raise gr.Error("Expected column 'IDfg' not found in FanGraphs table returned.")

    pt_col = "PA" if player_type == "Hitters" else "IP"
    if pt_col in df.columns and min_playing_time > 0:
        df = df[df[pt_col] >= min_playing_time].copy()

    if not target_col or target_col not in df.columns:
        target_col = pick_default_target(player_type, df)
        if not target_col:
            raise gr.Error("Could not find a numeric target column to use.")

    df = df.sort_values(["IDfg", "Season"]).copy()
    df["target_next"] = df.groupby("IDfg")[target_col].shift(-1)
    df = df.dropna(subset=["target_next"]).copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_cols if c not in ["target_next", "Season"]]

    if len(numeric_features) < 5:
        raise gr.Error("Not enough numeric feature columns found after filtering.")

    train = df[df["Season"] <= train_end].copy()
    val = df[df["Season"] == val_season].copy()
    test = df[df["Season"] == test_season].copy()

    if len(train) < 100:
        raise gr.Error(f"Train split too small (n={len(train)}). Expand season range or lower playing-time filter.")
    if len(test) < 20:
        raise gr.Error(f"Test split too small (n={len(test)}). Choose a different test season or expand season range.")

    X_train, y_train = train[numeric_features], train["target_next"]
    X_val, y_val = (val[numeric_features], val["target_next"]) if len(val) else (None, None)
    X_test, y_test = test[numeric_features], test["target_next"]

    model = build_model(model_name, numeric_features)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    lines = [
        f"### Results ({player_type})",
        f"- Target: **next-season {target_col}**",
        f"- Model: **{model_name}**",
        "",
        "#### Test metrics",
        f"- Season: **{test_season}**",
        f"- RMSE: **{rmse(y_test, test_pred):.4f}**",
        f"- MAE: **{mean_absolute_error(y_test, test_pred):.4f}**",
        f"- R²: **{safe_r2(y_test, test_pred):.4f}**",
        f"- n: **{len(test)}**",
    ]

    if len(val):
        val_pred = model.predict(X_val)
        lines += [
            "",
            "#### Validation metrics",
            f"- Season: **{val_season}**",
            f"- RMSE: **{rmse(y_val, val_pred):.4f}**",
            f"- MAE: **{mean_absolute_error(y_val, val_pred):.4f}**",
            f"- R²: **{safe_r2(y_val, val_pred):.4f}**",
            f"- n: **{len(val)}**",
        ]

    out = test[["Season"]].copy()
    if "Name" in test.columns:
        out["Name"] = test["Name"].values
    out["Predicted"] = test_pred
    out["Actual"] = y_test.values
    out["Error"] = out["Predicted"] - out["Actual"]
    out = out.sort_values("Predicted", ascending=False).head(int(top_n))

    return "\n".join(lines), out


with gr.Blocks(title="MLB Next-Season Predictor") as demo:
    gr.Markdown(
        "# ⚾ MLB Next-Season Predictor\n"
        "FanGraphs player-season data via **pybaseball** → next-season target → model → top predictions."
    )

    with gr.Row():
        player_type = gr.Dropdown(["Hitters", "Pitchers"], value="Hitters", label="Player type")
        model_name = gr.Dropdown(["Ridge", "Random Forest", "Gradient Boosting"], value="Ridge", label="Model")

    with gr.Row():
        start_season = gr.Number(value=2015, precision=0, label="Start season")
        end_season = gr.Number(value=2024, precision=0, label="End season")
        min_pt = gr.Number(value=200, precision=0, label="Min PA (hitters) / IP (pitchers)")

    target_col = gr.Textbox(
        value="",
        label="Target stat (leave blank for default)",
        placeholder="e.g., wOBA (hitters) or FIP (pitchers)"
    )

    with gr.Row():
        train_end = gr.Number(value=2021, precision=0, label="Train end season")
        val_season = gr.Number(value=2022, precision=0, label="Validation season")
        test_season = gr.Number(value=2023, precision=0, label="Test season")

    top_n = gr.Slider(5, 100, value=25, step=5, label="Top N predictions to show")
    run_btn = gr.Button("Run", variant="primary")

    summary = gr.Markdown()
    table = gr.Dataframe(interactive=False)

    def run(player_type, start_season, end_season, min_pt, target_col, model_name, train_end, val_season, test_season, top_n):
        start_season = int(start_season)
        end_season = int(end_season)
        min_pt = int(min_pt)
        train_end = int(train_end)
        val_season = int(val_season)
        test_season = int(test_season)

        if end_season < start_season:
            raise gr.Error("End season must be >= start season.")
        if test_season > end_season or test_season < start_season:
            raise gr.Error("Test season must be within your start/end range.")
        if val_season > end_season or val_season < start_season:
            raise gr.Error("Validation season must be within your start/end range.")
        if train_end >= test_season:
            raise gr.Error("Train end season should be before the test season.")

        return train_and_predict(
            player_type=player_type,
            start_season=start_season,
            end_season=end_season,
            min_playing_time=min_pt,
            target_col=target_col.strip(),
            model_name=model_name,
            train_end=train_end,
            val_season=val_season,
            test_season=test_season,
            top_n=int(top_n)
        )

    run_btn.click(
        run,
        inputs=[player_type, start_season, end_season, min_pt, target_col, model_name, train_end, val_season, test_season, top_n],
        outputs=[summary, table]
    )

if __name__ == "__main__":
    demo.launch()

    plt.scatter(y_test, preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    st.pyplot(fig)
