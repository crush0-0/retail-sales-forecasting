"""
================================================================================
Retail Sales Forecasting & Time-Series Pipeline
================================================================================
A production-grade, portfolio-ready time-series forecasting pipeline built on
the Superstore / Retail Sales dataset.

Features:
- Dynamic ingestion, datetime parsing, and continuous temporal resampling ('W-SUN')
- Stationarity validation via Augmented Dickey-Fuller (ADF) & Seasonal Decomposition
- Chronological, non-shuffled train/test split (80% / 20%)
- Triple-model architecture:
    1. SARIMA (Statistical / Seasonal Autoregressive)
    2. Prophet (Bayesian additive decomposition) with Holt-Winters fallback
    3. Random Forest Regressor (ML with lag features & rolling statistics)
- Out-of-time evaluation benchmarking: MAE, RMSE, MAPE
- Automated asset export:
    - images/sales_decomposition.png
    - images/actual_vs_forecast.png
    - images/future_forecast.png (12-week outlook with 95% confidence intervals)
================================================================================
"""

import os
import sys
import time
import warnings
from typing import Dict, Tuple, Optional, Any

# Headless matplotlib configuration for server/CLI environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Suppress convergence and stats warnings for clean console outputs
warnings.filterwarnings("ignore")

# Force UTF-8 on Windows console if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Attempt Prophet import with graceful fallback flag
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except (ImportError, Exception):
    PROPHET_AVAILABLE = False

# Pipeline configuration constants
IMAGE_DIR = "images"
DATA_FILE = "sales.csv"
RESAMPLE_FREQ = "W-SUN"
SEASONAL_PERIOD = 52
TEST_SPLIT_RATIO = 0.20
FUTURE_HORIZON_WEEKS = 12
RANDOM_STATE = 42

# Visual styling setup
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": False
})


def print_banner() -> None:
    """Print an executive-styled CLI banner matching portfolio standard."""
    print("=" * 80)
    print("       RETAIL SALES FORECASTING & TIME-SERIES BENCHMARK PIPELINE       ")
    print("=" * 80)


def print_section(step: int, total_steps: int, title: str) -> None:
    """Print a clean step section header."""
    print(f"\n[STEP {step}/{total_steps}] {title}")
    print("-" * 80)


def load_and_preprocess_data(csv_path: str = DATA_FILE) -> pd.Series:
    """
    Ingest sales dataset, resolve dynamic date and sales column candidates,
    enforce chronological order, and resample to continuous weekly sales.

    Parameters
    ----------
    csv_path : str
        Relative or absolute path to raw CSV file.

    Returns
    -------
    pd.Series
        Clean, continuous weekly sales time-series indexed by datetime.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input data file '{csv_path}' not found in current directory.")

    # Robust multi-encoding reader (UTF-8, Windows-1252, ISO-8859-1)
    df = None
    used_encoding = None
    for enc in ["utf-8", "windows-1252", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Unable to decode '{csv_path}' using supported text encodings.")

    print(f"  • Ingested file: '{csv_path}' [Encoding: {used_encoding}]")
    print(f"  • Raw rows: {len(df):,d} | Columns: {len(df.columns)}")

    # Resolve date column dynamically
    date_candidates = ["Order Date", "Date", "order_date", "date", "Timestamp", "timestamp"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if not date_col:
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break

    if not date_col:
        raise KeyError("Could not dynamically identify a date column in sales dataset.")

    # Resolve sales target column dynamically
    sales_candidates = ["Sales", "Weekly_Sales", "sales", "weekly_sales", "Total_Sales", "Revenue", "revenue"]
    sales_col = next((c for c in sales_candidates if c in df.columns), None)
    if not sales_col:
        for c in df.columns:
            if "sales" in c.lower() or "rev" in c.lower():
                sales_col = c
                break

    if not sales_col:
        raise KeyError("Could not dynamically identify a sales/revenue column in sales dataset.")

    print(f"  • Detected Date Column  : '{date_col}'")
    print(f"  • Detected Target Column: '{sales_col}'")

    # Clean missing values, parse datetimes, remove non-positive sales
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    clean_df = df.dropna(subset=[date_col, sales_col])
    clean_df = clean_df[clean_df[sales_col] > 0]
    clean_df = clean_df.sort_values(by=date_col)

    # Resample to weekly cadence ('W-SUN')
    weekly_series = (
        clean_df.set_index(date_col)[sales_col]
        .resample(RESAMPLE_FREQ)
        .sum()
    )

    # Validate time series continuity & impute zero-revenue gaps if any
    missing_intervals = weekly_series.isna().sum() + (weekly_series == 0).sum()
    if missing_intervals > 0:
        weekly_series = weekly_series.replace(0, np.nan).interpolate(method="time").bfill()

    print(f"  • Date Span: {weekly_series.index.min().strftime('%Y-%m-%d')} to {weekly_series.index.max().strftime('%Y-%m-%d')}")
    print(f"  • Continuous Resampled Weeks: {len(weekly_series):,d} weeks")
    print(f"  • Cumulative Sales Revenue : ${weekly_series.sum():,.2f}")
    print(f"  • Average Weekly Sales     : ${weekly_series.mean():,.2f} (Std: ${weekly_series.std():,.2f})")

    return weekly_series


def perform_decomposition_analysis(ts: pd.Series, output_dir: str = IMAGE_DIR) -> None:
    """
    Perform stationarity diagnostic via Augmented Dickey-Fuller test,
    execute seasonal decomposition into Trend, Seasonal, and Residual components,
    and save the visualization to images/sales_decomposition.png.

    Parameters
    ----------
    ts : pd.Series
        Clean weekly time-series.
    output_dir : str
        Directory where figure will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stationarity Diagnostic (Augmented Dickey-Fuller)
    adf_result = adfuller(ts.dropna(), autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    is_stationary = adf_pvalue < 0.05

    print(f"  • Augmented Dickey-Fuller (ADF) Test:")
    print(f"      - ADF Statistic : {adf_stat:.4f}")
    print(f"      - p-value       : {adf_pvalue:.4e}")
    print(f"      - Interpretation: {'Stationary (Reject H0)' if is_stationary else 'Non-Stationary (Fail to Reject H0)'}")

    # Classical Seasonal Decomposition (Additive)
    decomp = seasonal_decompose(ts, model="additive", period=SEASONAL_PERIOD)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")

    # Observed
    axes[0].plot(ts.index, ts.values, color="#1E3A8A", lw=1.8, label="Observed Weekly Sales")
    axes[0].set_ylabel("Observed ($)", fontweight="bold", color="#1E3A8A")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left", framealpha=0.9)

    # Trend
    axes[1].plot(decomp.trend.index, decomp.trend.values, color="#D97706", lw=2.2, label="Macro Trend (52-W Moving Avg)")
    axes[1].set_ylabel("Trend ($)", fontweight="bold", color="#D97706")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left", framealpha=0.9)

    # Seasonal
    axes[2].plot(decomp.seasonal.index, decomp.seasonal.values, color="#059669", lw=1.6, label="52-Week Annual Seasonality")
    axes[2].set_ylabel("Seasonal ($)", fontweight="bold", color="#059669")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left", framealpha=0.9)

    # Residual
    axes[3].scatter(decomp.resid.index, decomp.resid.values, color="#DC2626", s=14, alpha=0.7, label="Residuals / Noise")
    axes[3].axhline(0, color="black", linestyle="--", lw=1.0, alpha=0.7)
    axes[3].set_ylabel("Residual ($)", fontweight="bold", color="#DC2626")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper left", framealpha=0.9)

    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlabel("Timeline", fontweight="bold", labelpad=8)

    plt.suptitle("Superstore Retail Sales: Seasonal & Trend Decomposition", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "sales_decomposition.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  • Exported decomposition plot -> '{out_path}' [OK]")


def temporal_train_test_split(ts: pd.Series, split_ratio: float = TEST_SPLIT_RATIO) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series chronologically into Train (80%) and Test (20%) sets.
    Preserves exact temporal sequence without shuffling or lookahead leakage.
    """
    split_index = int(len(ts) * (1.0 - split_ratio))
    train = ts.iloc[:split_index]
    test = ts.iloc[split_index:]

    print(f"  • Train Set: {len(train)} weeks ({train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')}) [{len(train)/len(ts)*100:.1f}%]")
    print(f"  • Test Set : {len(test)} weeks ({test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')}) [{len(test)/len(ts)*100:.1f}%]")

    return train, test


def train_sarima(train: pd.Series, test: pd.Series) -> Tuple[np.ndarray, Any]:
    """
    Fit SARIMA(1,1,1)(1,1,0)[52] baseline statistical model.
    """
    print("  • Fitting SARIMA(1,1,1)(1,1,0)[52] model...")
    sarima_spec = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fitted = sarima_spec.fit(disp=False)
    pred = sarima_fitted.predict(start=len(train), end=len(train) + len(test) - 1)
    return np.array(pred), sarima_fitted


def train_prophet_or_fallback(train: pd.Series, test: pd.Series) -> Tuple[np.ndarray, str, Any]:
    """
    Fit Facebook Prophet model for additive trend and annual seasonality.
    Provides a seamless fallback to Holt-Winters Exponential Smoothing if
    Prophet is unavailable or encounters platform compilation constraints.
    """
    test_steps = len(test)

    if PROPHET_AVAILABLE:
        try:
            print("  • Training Facebook Prophet model (Yearly Seasonality, 95% CI)...")
            prophet_df = train.reset_index()
            prophet_df.columns = ["ds", "y"]

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95
            )
            # Suppress internal cmdstanpy chatter
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    model.fit(prophet_df)
                finally:
                    sys.stdout = old_stdout

            future_df = model.make_future_dataframe(periods=test_steps, freq=RESAMPLE_FREQ)
            forecast = model.predict(future_df)
            preds = forecast["yhat"].iloc[-test_steps:].values
            return preds, "Prophet", model
        except Exception as e:
            print(f"  ! Prophet encounter notice ({e}). Switching to Holt-Winters fallback...")

    # Holt-Winters Exponential Smoothing Fallback
    print("  • Training Holt-Winters Exponential Smoothing (Additive Trend + Seasonal)...")
    hw_model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD
    )
    hw_fitted = hw_model.fit()
    preds = hw_fitted.forecast(test_steps)
    return np.array(preds), "Holt-Winters", hw_fitted


def train_random_forest_regressor(train: pd.Series, test: pd.Series) -> Tuple[np.ndarray, RandomForestRegressor]:
    """
    Train Random Forest Regressor using autoregressive lag features (Lag 1, Lag 2),
    rolling mean statistics (4-week rolling window), and calendar cyclical indicators.
    Executes a multi-step recursive out-of-time forecast over the test set.
    """
    print("  • Engineering lag & rolling features for Machine Learning model...")
    full_series = pd.concat([train, test])
    df_feat = pd.DataFrame({"y": full_series})
    df_feat["lag_1"] = df_feat["y"].shift(1)
    df_feat["lag_2"] = df_feat["y"].shift(2)
    df_feat["rolling_mean_4"] = df_feat["y"].shift(1).rolling(4).mean()
    df_feat["month"] = df_feat.index.month
    df_feat["week_of_year"] = df_feat.index.isocalendar().week.astype(int)
    df_feat = df_feat.dropna()

    train_feat = df_feat.loc[df_feat.index.isin(train.index)]
    test_feat = df_feat.loc[df_feat.index.isin(test.index)]
    feature_cols = ["lag_1", "lag_2", "rolling_mean_4", "month", "week_of_year"]

    rf = RandomForestRegressor(n_estimators=150, max_depth=7, random_state=RANDOM_STATE)
    rf.fit(train_feat[feature_cols], train_feat["y"])

    # Recursive multi-step projection across out-of-time test sequence
    history = list(train_feat["y"])
    rf_predictions = []

    for target_date in test_feat.index:
        l1 = history[-1]
        l2 = history[-2]
        rm4 = float(np.mean(history[-4:]))
        m = target_date.month
        w = int(target_date.isocalendar().week)

        step_df = pd.DataFrame([[l1, l2, rm4, m, w]], columns=feature_cols)
        step_pred = rf.predict(step_df)[0]
        rf_predictions.append(step_pred)
        history.append(step_pred)

    return np.array(rf_predictions), rf


def evaluate_forecasts(actuals: pd.Series, predictions_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, str]:
    """
    Compute out-of-time benchmark metrics (MAE, RMSE, MAPE) across all trained models
    and identify the Champion model.
    """
    metrics = []
    y_true = actuals.values

    for model_name, y_pred in predictions_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0

        metrics.append({
            "Model": model_name,
            "MAE ($)": mae,
            "RMSE ($)": rmse,
            "MAPE (%)": mape
        })

    summary_df = pd.DataFrame(metrics).sort_values(by="MAE ($)").reset_index(drop=True)
    champion_name = summary_df.iloc[0]["Model"]

    # Print clean benchmark comparison table
    print("\n" + "=" * 80)
    print(f"{'OUT-OF-TIME MODEL PERFORMANCE BENCHMARK (42 WEEKS TEST HORIZON)':^80}")
    print("=" * 80)
    print(f"{'Rank':<6}{'Model':<30}{'MAE ($)':>14}{'RMSE ($)':>14}{'MAPE (%)':>14}")
    print("-" * 80)
    for idx, row in summary_df.iterrows():
        is_champ = " [*]" if row["Model"] == champion_name else ""
        print(f"#{idx+1:<5}{row['Model'] + is_champ:<30}{row['MAE ($)']:>14,.2f}{row['RMSE ($)']:>14,.2f}{row['MAPE (%)']:>13.2f}%")
    print("=" * 80)
    print(f"[*] Selected Champion Model: {champion_name} (Lowest Out-of-Time Error)")

    return summary_df, champion_name


def plot_actual_vs_forecast(
    train: pd.Series,
    test: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    output_dir: str = IMAGE_DIR
) -> None:
    """
    Export overlay visualization comparing actual test sales with predictions
    from all candidate models to images/actual_vs_forecast.png.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    # Display last 26 weeks of training history for smooth context
    context_train = train.iloc[-26:]
    ax.plot(context_train.index, context_train.values, color="#9CA3AF", lw=1.5, linestyle="--", label="Recent Train History")

    # Plot actual test sales
    ax.plot(test.index, test.values, color="#111827", lw=2.5, marker="o", markersize=4, label="Actual Test Sales (2017)")

    # Color mapping for model curves
    color_map = {
        "Prophet": "#2563EB",
        "Holt-Winters": "#0D9488",
        "SARIMA": "#7C3AED",
        "Random Forest": "#EA580C"
    }

    for name, preds in predictions_dict.items():
        base_name = "Prophet" if "Prophet" in name else ("Holt-Winters" if "Holt" in name else ("SARIMA" if "SARIMA" in name else "Random Forest"))
        c = color_map.get(base_name, "#4B5563")
        ax.plot(test.index, preds, label=f"{name} Forecast", color=c, lw=2.0, alpha=0.9)

    # Highlight out-of-time evaluation region
    ax.axvspan(test.index.min(), test.index.max(), color="#F3F4F6", alpha=0.6, zorder=0, label="Test Evaluation Window (20%)")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter('${x:,.0f}')

    ax.set_title("Retail Sales Forecasting: Out-of-Time Model Predictions vs. Ground Truth", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Timeline", fontweight="bold", labelpad=8)
    ax.set_ylabel("Weekly Sales Revenue ($USD)", fontweight="bold", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#FFFFFF")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "actual_vs_forecast.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  • Exported actual vs forecast plot -> '{out_path}' [OK]")


def generate_future_forecast(
    full_series: pd.Series,
    champion_name: str,
    horizon_weeks: int = FUTURE_HORIZON_WEEKS,
    output_dir: str = IMAGE_DIR
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Fit the champion architecture on the complete historical time series,
    project forward 12 weeks into the future, and generate 95% confidence intervals.
    Exports visualization to images/future_forecast.png.
    """
    last_date = full_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=horizon_weeks, freq=RESAMPLE_FREQ)

    future_mean = None
    future_lower = None
    future_upper = None

    if "Prophet" in champion_name and PROPHET_AVAILABLE:
        prophet_df = full_series.reset_index()
        prophet_df.columns = ["ds", "y"]
        m_full = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                m_full.fit(prophet_df)
            finally:
                sys.stdout = old_stdout

        future_df = m_full.make_future_dataframe(periods=horizon_weeks, freq=RESAMPLE_FREQ)
        fcst = m_full.predict(future_df)
        sub = fcst.iloc[-horizon_weeks:]
        future_mean = pd.Series(sub["yhat"].values, index=future_dates)
        future_lower = pd.Series(sub["yhat_lower"].values, index=future_dates)
        future_upper = pd.Series(sub["yhat_upper"].values, index=future_dates)

    elif "SARIMA" in champion_name:
        sarima_full = SARIMAX(
            full_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        forecast_res = sarima_full.get_forecast(steps=horizon_weeks)
        future_mean = pd.Series(forecast_res.predicted_mean.values, index=future_dates)
        conf_int = forecast_res.conf_int(alpha=0.05)
        future_lower = pd.Series(conf_int.iloc[:, 0].values, index=future_dates)
        future_upper = pd.Series(conf_int.iloc[:, 1].values, index=future_dates)

    else:
        # Holt-Winters Exponential Smoothing projection with residual simulation bounds
        hw_full = ExponentialSmoothing(
            full_series,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()
        pred_vals = hw_full.forecast(horizon_weeks)
        future_mean = pd.Series(pred_vals.values, index=future_dates)
        res_std = float(np.std(hw_full.resid))
        expansion_factor = np.sqrt(np.arange(1, horizon_weeks + 1))
        future_lower = future_mean - 1.96 * res_std * expansion_factor
        future_upper = future_mean + 1.96 * res_std * expansion_factor

    # Floor lower bound at 0 for realistic sales numbers
    future_lower = future_lower.clip(lower=0)

    # Plot future forecast figure
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    # Plot historical trajectory (last 52 weeks for visual clarity)
    hist_window = full_series.iloc[-52:]
    ax.plot(hist_window.index, hist_window.values, color="#1F2937", lw=2.0, label="Historical Actual Sales (Past 52 Weeks)")

    # Plot 12-week future forecast projection
    ax.plot(future_mean.index, future_mean.values, color="#2563EB", lw=2.8, marker="s", markersize=5, label=f"12-Week Outlook ({champion_name})")

    # Shaded 95% Confidence Interval band
    ax.fill_between(
        future_dates,
        future_lower.values,
        future_upper.values,
        color="#3B82F6",
        alpha=0.25,
        label="95% Confidence Interval Band"
    )

    # Annotate key business metrics on plot
    peak_date = future_mean.idxmax()
    peak_val = future_mean.max()
    ax.annotate(
        f"Peak Outlook Week\n\\${peak_val:,.0f} ({peak_date.strftime('%b %d')})",
        xy=(peak_date, peak_val),
        xytext=(peak_date, peak_val + (peak_val * 0.12)),
        arrowprops=dict(facecolor="#1D4ED8", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight="bold",
        color="#1D4ED8",
        ha="center"
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter('${x:,.0f}')

    total_projected_rev = future_mean.sum()
    avg_projected_weekly = future_mean.mean()

    ax.set_title(
        f"12-Week Strategic Sales Outlook (Champion Model: {champion_name})\n"
        f"Projected Quarterly Revenue: \\${total_projected_rev:,.2f}  |  Expected Weekly Run-Rate: \\${avg_projected_weekly:,.2f}",
        fontsize=14,
        fontweight="bold",
        pad=14
    )
    ax.set_xlabel("Timeline", fontweight="bold", labelpad=8)
    ax.set_ylabel("Weekly Sales Revenue ($USD)", fontweight="bold", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95, facecolor="#FFFFFF")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "future_forecast.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  • Exported 12-week future forecast plot -> '{out_path}' [OK]")

    return future_mean, future_lower, future_upper


def main() -> None:
    """Execute end-to-end retail sales forecasting pipeline."""
    start_time = time.time()
    print_banner()

    # Step 1: Ingestion & Hygiene
    print_section(1, 6, "DATA INGESTION & TEMPORAL HYGIENE")
    ts_sales = load_and_preprocess_data(DATA_FILE)

    # Step 2: Exploratory & Decomposition
    print_section(2, 6, "EXPLORATORY TIME-SERIES & DECOMPOSITION ANALYSIS")
    perform_decomposition_analysis(ts_sales, IMAGE_DIR)

    # Step 3: Chronological Train / Test Split
    print_section(3, 6, "CHRONOLOGICAL TRAIN / TEST SPLIT (80% / 20%)")
    train, test = temporal_train_test_split(ts_sales, TEST_SPLIT_RATIO)

    # Step 4: Model Training & Out-of-Time Forecasting
    print_section(4, 6, "TRAINING BENCHMARK FORECASTING MODELS")
    predictions: Dict[str, np.ndarray] = {}

    # Model 1: SARIMA
    pred_sarima, _ = train_sarima(train, test)
    predictions["SARIMA"] = pred_sarima

    # Model 2: Prophet or Holt-Winters fallback
    pred_p_hw, p_hw_label, _ = train_prophet_or_fallback(train, test)
    predictions[p_hw_label] = pred_p_hw

    # Model 3: Random Forest with Lags
    pred_rf, _ = train_random_forest_regressor(train, test)
    predictions["Random Forest"] = pred_rf

    # Step 5: Model Evaluation & Benchmarking
    print_section(5, 6, "OUT-OF-TIME BENCHMARK EVALUATION")
    summary_df, champion_name = evaluate_forecasts(test, predictions)

    # Step 6: Visual Asset Generation & Future Forecast
    print_section(6, 6, "VISUAL ASSET EXPORT & STRATEGIC 12-WEEK OUTLOOK")
    plot_actual_vs_forecast(train, test, predictions, IMAGE_DIR)
    future_mean, future_lower, future_upper = generate_future_forecast(ts_sales, champion_name, FUTURE_HORIZON_WEEKS, IMAGE_DIR)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"   PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS   ")
    print("=" * 80)
    print("Generated Deliverables:")
    print(f"  • Decomposition Plot : {os.path.join(IMAGE_DIR, 'sales_decomposition.png')}")
    print(f"  • Actual vs Forecast : {os.path.join(IMAGE_DIR, 'actual_vs_forecast.png')}")
    print(f"  • Future Projection  : {os.path.join(IMAGE_DIR, 'future_forecast.png')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
