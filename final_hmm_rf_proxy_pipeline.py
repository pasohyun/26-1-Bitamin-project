import argparse
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


HMM_FEATURES = [
    "Equity_vs_Bond",
    "US10Y",
    "Jobless_Claims_MA",
    "VIX",
    "Market_Breadth",
    "spread_10y2y",
    "DXY",
    "Copper_Gold_Ratio",
]

RF_23_FEATURES = [
    "Equity_vs_Bond",
    "US10Y",
    "DXY",
    "Market_Breadth",
    "VIX",
    "Jobless_Claims_MA",
    "spread_10y2y",
    "Copper_Gold_Ratio",
    "MA200_Dist",
    "Month",
    "Vol_Ratio",
    "SPY_MA20_Diff",
    "SPY_Log_Ret",
    "SPY_ret",
    "VIX_MA5_Diff",
    "RSI",
    "VIX_ret",
    "HY_spread_ret",
    "OIL_ret",
    "GOLD_ret",
    "USDKRW_ret",
    "DXY_ret",
    "US10Y_diff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final HMM 6-regime visualization and RF proxy ranking from the full merged dataset."
    )
    parser.add_argument("--cache", default="cached_raw_data.csv", help="Merged raw data CSV.")
    parser.add_argument("--outdir", default="final_hmm_rf_outputs", help="Directory for charts and CSV outputs.")
    parser.add_argument("--start", default="2010-01-01", help="Analysis start date.")
    parser.add_argument("--seed", type=int, default=42, help="GaussianHMM random_state.")
    parser.add_argument(
        "--equity-bond-mode",
        choices=["target_style", "true_bond"],
        default="target_style",
        help=(
            "target_style uses SPY/EEM to reproduce the existing screenshots' scale. "
            "true_bond uses SPY/AGG if present, otherwise SPY/IEF."
        ),
    )
    return parser.parse_args()


def configure_korean_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_name = fm.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_raw_cache(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    return raw.ffill().bfill()


def choose_equity_bond_denominator(raw: pd.DataFrame, mode: str) -> str:
    if mode == "target_style":
        if "EEM" not in raw.columns:
            raise ValueError("target_style needs EEM in the raw data.")
        return "EEM"

    if "AGG" in raw.columns:
        return "AGG"
    if "IEF" in raw.columns:
        return "IEF"
    raise ValueError("true_bond mode needs AGG or IEF in the raw data.")


def calc_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(raw: pd.DataFrame, denominator: str) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    spy_ret = raw["SPY"].pct_change()

    df["SPY"] = raw["SPY"]
    df["Equity_vs_Bond"] = raw["SPY"] / raw[denominator]
    df["VIX"] = raw["VIX"]
    df["VIX_ret"] = raw["VIX"].pct_change()
    df["VIX_MA5_Diff"] = raw["VIX"] - raw["VIX"].rolling(5).mean()
    df["Volatility_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
    df["Vol_Ratio"] = df["Volatility_20d"] / (spy_ret.rolling(60).std() * np.sqrt(252))

    df["DXY"] = raw["DXY"]
    df["DXY_ret"] = raw["DXY"].pct_change()
    df["USDKRW_ret"] = raw["USDKRW"].pct_change() if "USDKRW" in raw.columns else np.nan

    df["US10Y"] = raw["US10Y"]
    df["US10Y_diff"] = raw["US10Y"].diff()
    df["spread_10y2y"] = raw["US10Y"] - raw["US2Y"]

    df["Market_Breadth"] = raw["RSP"] / raw["SPY"]
    df["Copper_Gold_Ratio"] = raw["Copper"] / raw["Gold"]
    df["GOLD_ret"] = raw["Gold"].pct_change()
    oil_col = "WTI_Oil" if "WTI_Oil" in raw.columns else "CL=F"
    df["OIL_ret"] = raw[oil_col].pct_change()

    if "HY_Spread" in raw.columns:
        df["HY_spread_ret"] = raw["HY_Spread"].diff()
    elif {"HYG", "AGG"}.issubset(raw.columns):
        df["HY_spread_ret"] = (raw["HYG"] / raw["AGG"]).diff()
    else:
        df["HY_spread_ret"] = np.nan

    df["SPY_ret"] = spy_ret
    df["SPY_Log_Ret"] = np.log(raw["SPY"] / raw["SPY"].shift(1))
    df["SPY_MA20_Diff"] = (raw["SPY"] / raw["SPY"].rolling(20).mean()) - 1
    df["MA200_Dist"] = raw["SPY"] / raw["SPY"].rolling(200).mean()
    df["RSI"] = calc_rsi(raw["SPY"])
    df["Month"] = df.index.month
    df["Jobless_Claims_MA"] = raw["Jobless_Claims"].rolling(20).mean()
    return df


def fit_hmm(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    scaler = StandardScaler()
    x_hmm = scaler.fit_transform(df[HMM_FEATURES])

    hmm = GaussianHMM(n_components=6, covariance_type="full", n_iter=1000, tol=1e-3, random_state=seed)
    hmm.fit(x_hmm)
    states = hmm.predict(x_hmm)

    result = df.copy()
    result["Regime"] = states

    # This is the same level mapping used in HMM_Final.ipynb: lower VIX means safer level.
    sorted_regimes = result.groupby("Regime")["VIX"].mean().sort_values().index.tolist()
    rank_map = {regime: rank + 1 for rank, regime in enumerate(sorted_regimes)}
    result["Level"] = result["Regime"].map(rank_map)
    return result


def train_rf_proxy(labeled: pd.DataFrame) -> pd.DataFrame:
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
        oob_score=True,
    )
    rf.fit(labeled[RF_23_FEATURES], labeled["Level"])

    importance = pd.DataFrame(
        {
            "feature": RF_23_FEATURES,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance["importance_pct"] = importance["importance"] * 100
    importance.attrs["oob_score"] = rf.oob_score_
    return importance


def plot_hmm(labeled: pd.DataFrame, output_path: Path) -> None:
    colors = [plt.cm.RdYlGn_r(i / 5) for i in range(6)]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(labeled.index, labeled["SPY"], color="black", linewidth=1.2, label="SPY Price")

    for level in range(1, 7):
        condition = labeled["Level"] == level
        ax.fill_between(
            labeled.index,
            labeled["SPY"].min(),
            labeled["SPY"].max(),
            where=condition,
            color=colors[level - 1],
            alpha=0.5,
            label=f"Level {level}",
        )

    ax.set_title("[최종 마스터피스] HMM 6국면 시각화 (8개 핵심 센서)", fontsize=18, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", ncol=7, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rf_proxy(importance: pd.DataFrame, output_path: Path) -> None:
    plot_df = importance.sort_values("importance_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(plot_df["feature"], plot_df["importance_pct"], color="#349a9a")
    ax.set_title("Random Forest Proxy: 23개 센서 중요도 랭킹 (HMM 6국면 기준)", fontsize=16, fontweight="bold")
    ax.set_xlabel("중요도 기여율 (%)")
    ax.set_ylabel("퀀트 피처 (Sensors)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, plot_df["importance_pct"]):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)

    ax.set_xlim(0, max(plot_df["importance_pct"].max() * 1.15, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_korean_font()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_cache(Path(args.cache))
    denominator = choose_equity_bond_denominator(raw, args.equity_bond_mode)
    features = build_features(raw, denominator)
    needed = ["SPY", *HMM_FEATURES, *RF_23_FEATURES]
    features = features.loc[args.start :, needed]
    features = features.loc[:, ~features.columns.duplicated()]

    # Keep the HMM sample independent from the RF candidate feature set.
    # HMM_Final.ipynb learns regimes from only these 8 core sensors; dropping rows
    # because an RF-only sensor is missing can send GaussianHMM to a different local optimum.
    hmm_input = features[["SPY", *HMM_FEATURES]].dropna()
    labeled = fit_hmm(hmm_input, args.seed)

    rf_input = labeled.join(features[RF_23_FEATURES], how="left", rsuffix="_rf")
    rf_input = rf_input.loc[:, ~rf_input.columns.duplicated()].dropna(subset=RF_23_FEATURES)
    importance = train_rf_proxy(rf_input)

    prefix = f"{args.equity_bond_mode}_seed{args.seed}"
    hmm_png = outdir / f"hmm_6_regimes_{prefix}.png"
    rf_png = outdir / f"rf_proxy_23_sensors_{prefix}.png"
    stats_csv = outdir / f"hmm_level_means_{prefix}.csv"
    importance_csv = outdir / f"rf_proxy_importance_{prefix}.csv"
    labeled_csv = outdir / f"hmm_labeled_data_{prefix}.csv"

    level_means = labeled.groupby("Level")[HMM_FEATURES].mean()
    level_means.index = [f"Level {i}" for i in level_means.index]

    plot_hmm(labeled, hmm_png)
    plot_rf_proxy(importance, rf_png)
    level_means.to_csv(stats_csv, encoding="utf-8-sig")
    importance.to_csv(importance_csv, index=False, encoding="utf-8-sig")
    labeled[["SPY", "Regime", "Level", *HMM_FEATURES]].to_csv(labeled_csv, encoding="utf-8-sig")

    print("=" * 95)
    print("Final HMM + RF proxy pipeline complete")
    print("=" * 95)
    print(f"Input cache              : {Path(args.cache).resolve()}")
    print(f"HMM rows used            : {len(labeled):,}")
    print(f"RF rows used             : {len(rf_input):,}")
    print(f"Date range               : {labeled.index.min().date()} ~ {labeled.index.max().date()}")
    print(f"Equity_vs_Bond           : SPY / {denominator}")
    print(f"HMM seed                 : {args.seed}")
    print(f"RF OOB score             : {importance.attrs['oob_score']:.4f}")
    print()
    print("[Top 10 RF proxy sensors]")
    for _, row in importance.head(10).iterrows():
        print(f"{row['feature']:<22} {row['importance_pct']:>6.2f}%")
    print()
    print("[HMM level means]")
    print(level_means.round(3).to_string())
    print()
    print(f"Saved HMM chart          : {hmm_png.resolve()}")
    print(f"Saved RF proxy chart     : {rf_png.resolve()}")
    print(f"Saved level means        : {stats_csv.resolve()}")


if __name__ == "__main__":
    main()
