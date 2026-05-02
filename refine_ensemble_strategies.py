#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refined A2C + PPO ensemble strategy experiment.

This script does not retrain RL models. It loads the already trained
fold-level PPO/A2C models from ensemble_models/ and re-runs only the
ensemble-ratio logic.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import stable_baselines3 first for this Windows/conda environment.
from stable_baselines3 import A2C, PPO
from hmmlearn.hmm import GaussianHMM

import warnings
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CACHE_PATH = "cached_raw_data.csv"
MODEL_DIR = "ensemble_models"
RESULT_DIR = "ensemble_refined_results"
os.makedirs(RESULT_DIR, exist_ok=True)

FEATURES = [
    "Equity_vs_Bond",
    "US10Y",
    "Jobless_Claims_MA",
    "VIX",
    "Market_Breadth",
    "spread_10y2y",
    "DXY",
    "Copper_Gold",
    "SPY_Mom_3M",
]
ASSETS = ["SPY", "TLT", "SHV", "GLD", "DBC"]

HMM_N_STATES = 6
HMM_PROB_COLS = [f"HMM_Prob_{i}" for i in range(HMM_N_STATES)]
OBS_COLS = FEATURES + HMM_PROB_COLS

REBALANCE_FREQ = "B"
ANNUALIZATION = 252
SHARPE_WINDOW = 252
TRAIN_YEARS = 4
TEST_YEARS = 1
TRANSACTION_COST = 0.0005


def softmax_action(action):
    z = np.asarray(action, dtype=np.float64).reshape(-1)
    z = np.clip(z, -20, 20)
    exp_z = np.exp(z - np.max(z))
    return exp_z / (exp_z.sum() + 1e-12)


def prepare_data():
    raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True).sort_index()

    df = raw.copy()
    df["Equity_vs_Bond"] = df["SPY"] / df["EEM"]
    df["Copper_Gold"] = df["Copper"] / df["Gold"]
    df["spread_10y2y"] = df["US10Y"] - df["US2Y"]
    df["Market_Breadth"] = df["RSP"] / df["SPY"]
    df["Jobless_Claims_MA"] = df["Jobless_Claims"].rolling(20).mean()
    df["SPY_Mom_3M"] = df["SPY"].pct_change(60)

    feature_daily = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    period_prices = raw[ASSETS].resample(REBALANCE_FREQ).last()
    period_returns = period_prices.pct_change()
    period_features = feature_daily.resample(REBALANCE_FREQ).last().shift(1)
    period_raw_features = df[["VIX", "spread_10y2y", "SPY_Mom_3M"]].resample(REBALANCE_FREQ).last().shift(1)

    dataset = pd.concat(
        [
            period_features.add_prefix("x_"),
            period_raw_features.add_prefix("raw_"),
            period_returns.add_prefix("r_"),
        ],
        axis=1,
    ).dropna()

    x_all = dataset[[f"x_{c}" for c in FEATURES]].copy()
    x_all.columns = FEATURES
    raw_signal_all = dataset[[f"raw_{c}" for c in ["VIX", "spread_10y2y", "SPY_Mom_3M"]]].copy()
    raw_signal_all.columns = ["VIX", "spread_10y2y", "SPY_Mom_3M"]
    r_all = dataset[[f"r_{c}" for c in ASSETS]].copy()
    r_all.columns = ASSETS
    return dataset, x_all, raw_signal_all, r_all, feature_daily


def make_fold_hmm_prob_features(split, feature_daily):
    train_daily = feature_daily.loc[
        (feature_daily.index >= split["train_start"]) &
        (feature_daily.index <= split["train_end"])
    ].copy()
    inference_daily = feature_daily.loc[
        (feature_daily.index >= split["train_start"]) &
        (feature_daily.index <= split["test_end"])
    ].copy()

    hmm_scaler = StandardScaler()
    train_scaled = hmm_scaler.fit_transform(train_daily[FEATURES])
    inference_scaled = hmm_scaler.transform(inference_daily[FEATURES])

    hmm = GaussianHMM(n_components=HMM_N_STATES, covariance_type="full", n_iter=1000, random_state=42 + split["fold"])
    hmm.fit(train_scaled)

    train_states = hmm.predict(train_scaled)
    vix_by_state = pd.Series(train_daily["VIX"].values).groupby(train_states).mean()
    ordered_states = vix_by_state.sort_values().index.tolist()

    raw_probs = hmm.predict_proba(inference_scaled)
    ranked_probs = np.zeros_like(raw_probs)
    for new_rank, old_state in enumerate(ordered_states):
        ranked_probs[:, new_rank] = raw_probs[:, old_state]

    daily_probs = pd.DataFrame(ranked_probs, index=inference_daily.index, columns=HMM_PROB_COLS)
    return daily_probs.resample(REBALANCE_FREQ).last().shift(1)


def append_hmm_probs(x_test, split, feature_daily):
    period_probs = make_fold_hmm_prob_features(split, feature_daily)
    test_probs = period_probs.reindex(x_test.index)
    x_test_obs = pd.concat([x_test, test_probs], axis=1).dropna()
    return x_test_obs[OBS_COLS]


def make_walk_forward_splits(index):
    first_year = int(index.min().year)
    last_date = index.max()
    splits = []
    start_year = first_year

    while True:
        train_start = pd.Timestamp(f"{start_year}-01-01")
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=TEST_YEARS) - pd.DateOffset(days=1)

        if test_start > last_date:
            break
        if test_end > last_date:
            test_end = last_date

        train_mask = (index >= train_start) & (index <= train_end)
        test_mask = (index >= test_start) & (index <= test_end)
        if train_mask.sum() >= 24 and test_mask.sum() >= 6:
            splits.append(
                {
                    "fold": len(splits) + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                }
            )

        start_year += TEST_YEARS
        if test_end >= last_date:
            break

    return splits


def predict_weight_path(model, x_test):
    rows = []
    for obs in x_test.values.astype(np.float32):
        action, _ = model.predict(obs, deterministic=True)
        rows.append(softmax_action(action))
    return pd.DataFrame(rows, index=x_test.index, columns=ASSETS)


def rolling_sharpe_score(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    return float(arr.mean() / (arr.std() + 1e-8) * np.sqrt(ANNUALIZATION))


def ratio_from_scores(ppo_score, a2c_score, temperature=1.0):
    scores = np.array([ppo_score, a2c_score], dtype=np.float64) / temperature
    scores = np.clip(scores, -10, 10)
    exp_scores = np.exp(scores - scores.max())
    return float(exp_scores[0] / exp_scores.sum())


def apply_dd_cap(ppo_ratio, current_dd, strong=False):
    if not strong:
        return min(ppo_ratio, 0.25) if current_dd > 0.10 else ppo_ratio
    if current_dd > 0.20:
        return 0.0
    if current_dd > 0.15:
        return min(ppo_ratio, 0.10)
    if current_dd > 0.10:
        return min(ppo_ratio, 0.25)
    return ppo_ratio


def regime_risk_score(signal_row, thresholds):
    score = 0
    if signal_row["VIX"] > thresholds["vix_high"]:
        score += 1
    if signal_row["spread_10y2y"] < 0:
        score += 1
    if signal_row["SPY_Mom_3M"] < 0:
        score += 1
    return score


def get_strategy_ratio(
    strategy,
    dt,
    ppo_hist,
    a2c_hist,
    net_ret_hist,
    port_val,
    peak_val,
    prev_ratio,
    signals,
    thresholds,
):
    current_dd = (peak_val - port_val) / (peak_val + 1e-8)
    base_ratio = ratio_from_scores(rolling_sharpe_score(ppo_hist), rolling_sharpe_score(a2c_hist))
    kind = strategy["kind"]

    if kind == "fixed":
        return strategy["fixed_ppo"]

    if kind == "auto_sharpe":
        return base_ratio

    if kind == "auto_dd_guard_v1":
        return apply_dd_cap(base_ratio, current_dd, strong=False)

    if kind == "auto_dd_guard_v2":
        return apply_dd_cap(base_ratio, current_dd, strong=True)

    if kind == "regime_dd_guard":
        risk = regime_risk_score(signals.loc[dt], thresholds)
        target = base_ratio
        if risk == 0:
            target = max(target, 0.55)
        elif risk == 1:
            target = min(target, 0.50)
        elif risk == 2:
            target = min(target, 0.25)
        else:
            target = 0.0
        return apply_dd_cap(target, current_dd, strong=True)

    if kind == "turnover_aware_dd_guard":
        target = apply_dd_cap(base_ratio, current_dd, strong=True)
        if prev_ratio is None:
            return target
        if abs(target - prev_ratio) < 0.10:
            return prev_ratio
        return 0.75 * prev_ratio + 0.25 * target

    if kind == "vol_target_dd_guard":
        target = apply_dd_cap(base_ratio, current_dd, strong=True)
        if len(net_ret_hist) >= 6:
            trailing_vol = float(np.std(net_ret_hist) * np.sqrt(ANNUALIZATION))
            if trailing_vol > strategy["vol_target"]:
                target = min(target, target * strategy["vol_target"] / (trailing_vol + 1e-8))
        return target

    if kind == "regime_tactical":
        signal = signals.loc[dt]
        risk = regime_risk_score(signal, thresholds)
        if risk >= 2:
            target = 0.15
        elif risk == 0 and signal["VIX"] < thresholds["vix_mid"] and signal["SPY_Mom_3M"] > 0:
            target = 0.70
        else:
            target = 0.45
        return apply_dd_cap(target, current_dd, strong=True)

    raise ValueError(f"Unknown strategy kind: {kind}")


def run_ensemble_strategy(ppo_weights, a2c_weights, returns, signals, thresholds, strategy):
    ppo_hist = deque(maxlen=SHARPE_WINDOW)
    a2c_hist = deque(maxlen=SHARPE_WINDOW)
    net_ret_hist = deque(maxlen=SHARPE_WINDOW)

    prev_w = np.ones(len(ASSETS)) / len(ASSETS)
    prev_ratio = None
    port_val = 1.0
    peak_val = 1.0

    ratio_rows = []
    weight_rows = []
    net_returns = []
    turnovers = []
    costs = []
    dd_rows = []

    for dt in returns.index:
        ppo_w = ppo_weights.loc[dt].values
        a2c_w = a2c_weights.loc[dt].values
        r = returns.loc[dt].values

        ppo_ratio = get_strategy_ratio(
            strategy,
            dt,
            ppo_hist,
            a2c_hist,
            net_ret_hist,
            port_val,
            peak_val,
            prev_ratio,
            signals,
            thresholds,
        )
        ppo_ratio = float(np.clip(ppo_ratio, 0.0, 1.0))

        final_w = ppo_ratio * ppo_w + (1.0 - ppo_ratio) * a2c_w
        final_w = final_w / (final_w.sum() + 1e-12)

        ppo_model_ret = float(np.dot(ppo_w, r))
        a2c_model_ret = float(np.dot(a2c_w, r))
        gross_ret = float(np.dot(final_w, r))
        turnover = float(np.sum(np.abs(final_w - prev_w)) / 2.0)
        cost = turnover * TRANSACTION_COST
        net_ret = gross_ret - cost

        port_val *= 1.0 + net_ret
        peak_val = max(peak_val, port_val)
        current_dd = (peak_val - port_val) / (peak_val + 1e-8)

        ppo_hist.append(ppo_model_ret)
        a2c_hist.append(a2c_model_ret)
        net_ret_hist.append(net_ret)
        prev_w = final_w
        prev_ratio = ppo_ratio

        ratio_rows.append(ppo_ratio)
        weight_rows.append(final_w)
        net_returns.append(net_ret)
        turnovers.append(turnover)
        costs.append(cost)
        dd_rows.append(current_dd)

    return {
        "strategy": strategy["name"],
        "ppo_ratio": pd.Series(ratio_rows, index=returns.index, name="ppo_ratio"),
        "weights": pd.DataFrame(weight_rows, index=returns.index, columns=ASSETS),
        "net_returns": pd.Series(net_returns, index=returns.index, name="net_return"),
        "turnover": pd.Series(turnovers, index=returns.index, name="turnover"),
        "cost": pd.Series(costs, index=returns.index, name="cost"),
        "drawdown": pd.Series(dd_rows, index=returns.index, name="drawdown"),
    }


def max_drawdown(equity):
    eq = pd.Series(equity).dropna()
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def perf_metrics(returns, turnover=None):
    ret = pd.Series(returns).dropna()
    equity = (1.0 + ret).cumprod()
    years = len(ret) / ANNUALIZATION
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(ret.std() * np.sqrt(ANNUALIZATION))
    sharpe = float(ret.mean() / (ret.std() + 1e-8) * np.sqrt(ANNUALIZATION))
    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    avg_turnover = float(pd.Series(turnover).mean()) if turnover is not None else np.nan
    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "avg_turnover": avg_turnover,
    }


STRATEGIES = [
    {"name": "PPO_Only", "kind": "fixed", "fixed_ppo": 1.00},
    {"name": "A2C_Only", "kind": "fixed", "fixed_ppo": 0.00},
    {"name": "Manual_PPO25_A2C75", "kind": "fixed", "fixed_ppo": 0.25},
    {"name": "Manual_PPO50_A2C50", "kind": "fixed", "fixed_ppo": 0.50},
    {"name": "Manual_PPO75_A2C25", "kind": "fixed", "fixed_ppo": 0.75},
    {"name": "Auto_RollingSharpe", "kind": "auto_sharpe"},
    {"name": "Auto_DD_Guard_v1", "kind": "auto_dd_guard_v1"},
    {"name": "Auto_DD_Guard_v2_Strong", "kind": "auto_dd_guard_v2"},
    {"name": "Regime_DD_Guard", "kind": "regime_dd_guard"},
    {"name": "Turnover_Aware_DD_Guard", "kind": "turnover_aware_dd_guard"},
    {"name": "Vol_Target_DD_Guard", "kind": "vol_target_dd_guard", "vol_target": 0.10},
    {"name": "Regime_Tactical", "kind": "regime_tactical"},
]


def main():
    dataset, x_all, raw_signal_all, r_all, feature_daily = prepare_data()
    splits = make_walk_forward_splits(dataset.index)

    print(f"Loaded data: {dataset.index.min().date()} ~ {dataset.index.max().date()}")
    print(f"Folds: {len(splits)}")
    print("Running refined ensemble strategies with saved PPO/A2C models...")

    fold_rows = []
    oos_return_parts = {s["name"]: [] for s in STRATEGIES}
    oos_turnover_parts = {s["name"]: [] for s in STRATEGIES}
    oos_ratio_parts = {s["name"]: [] for s in STRATEGIES}
    oos_weight_parts = {s["name"]: [] for s in STRATEGIES}

    for split in splits:
        fold = split["fold"]
        ppo_path = os.path.join(MODEL_DIR, f"fold{fold}_ppo.zip")
        a2c_path = os.path.join(MODEL_DIR, f"fold{fold}_a2c.zip")
        if not os.path.exists(ppo_path) or not os.path.exists(a2c_path):
            raise FileNotFoundError(f"Missing saved models for fold {fold}: {ppo_path}, {a2c_path}")

        print(
            f"Fold {fold:02d}: "
            f"Train {split['train_start'].date()}~{split['train_end'].date()} | "
            f"Test {split['test_start'].date()}~{split['test_end'].date()}"
        )

        train_mask = split["train_mask"]
        test_mask = split["test_mask"]

        scaler = StandardScaler()
        x_train = pd.DataFrame(
            scaler.fit_transform(x_all.loc[train_mask]),
            index=x_all.loc[train_mask].index,
            columns=FEATURES,
        )
        x_test_base = pd.DataFrame(
            scaler.transform(x_all.loc[test_mask]),
            index=x_all.loc[test_mask].index,
            columns=FEATURES,
        )
        x_test = append_hmm_probs(x_test_base, split, feature_daily)
        del x_train

        r_test = r_all.loc[x_test.index]
        signals_train = raw_signal_all.loc[train_mask]
        signals_test = raw_signal_all.loc[x_test.index]
        thresholds = {
            "vix_high": float(signals_train["VIX"].quantile(0.80)),
            "vix_mid": float(signals_train["VIX"].quantile(0.50)),
        }

        ppo_model = PPO.load(ppo_path)
        a2c_model = A2C.load(a2c_path)
        ppo_weights = predict_weight_path(ppo_model, x_test)
        a2c_weights = predict_weight_path(a2c_model, x_test)

        for strategy in STRATEGIES:
            res = run_ensemble_strategy(ppo_weights, a2c_weights, r_test, signals_test, thresholds, strategy)
            metrics = perf_metrics(res["net_returns"], res["turnover"])
            fold_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy["name"],
                    "test_start": split["test_start"].date(),
                    "test_end": split["test_end"].date(),
                    **metrics,
                    "mean_ppo_ratio": float(res["ppo_ratio"].mean()),
                    "vix_high_threshold": thresholds["vix_high"],
                }
            )
            oos_return_parts[strategy["name"]].append(res["net_returns"])
            oos_turnover_parts[strategy["name"]].append(res["turnover"])
            oos_ratio_parts[strategy["name"]].append(res["ppo_ratio"])
            oos_weight_parts[strategy["name"]].append(res["weights"])

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(os.path.join(RESULT_DIR, "refined_fold_results.csv"), index=False, encoding="utf-8-sig")

    aggregate_rows = []
    equity_df = pd.DataFrame()
    ratio_df = pd.DataFrame()
    for strategy in [s["name"] for s in STRATEGIES]:
        ret = pd.concat(oos_return_parts[strategy]).sort_index()
        turnover = pd.concat(oos_turnover_parts[strategy]).sort_index()
        ratio = pd.concat(oos_ratio_parts[strategy]).sort_index()
        weights = pd.concat(oos_weight_parts[strategy]).sort_index()
        metrics = perf_metrics(ret, turnover)

        aggregate_rows.append(
            {
                "strategy": strategy,
                **metrics,
                "mean_ppo_ratio": float(ratio.mean()),
            }
        )
        equity_df[strategy] = (1.0 + ret).cumprod()
        ratio_df[strategy] = ratio
        weights.to_csv(os.path.join(RESULT_DIR, f"weights_{strategy}.csv"), encoding="utf-8-sig")

    summary = pd.DataFrame(aggregate_rows).sort_values(["sharpe", "calmar"], ascending=False)
    summary.to_csv(os.path.join(RESULT_DIR, "refined_summary.csv"), index=False, encoding="utf-8-sig")
    equity_df.to_csv(os.path.join(RESULT_DIR, "refined_oos_equity_curves.csv"), encoding="utf-8-sig")
    ratio_df.to_csv(os.path.join(RESULT_DIR, "refined_oos_ppo_ratios.csv"), encoding="utf-8-sig")

    best_strategy = summary.iloc[0]["strategy"]
    print("\n=== Refined Summary ===")
    print(
        summary.assign(
            total_return=lambda x: x["total_return"] * 100,
            cagr=lambda x: x["cagr"] * 100,
            vol=lambda x: x["vol"] * 100,
            mdd=lambda x: x["mdd"] * 100,
        ).round(3).to_string(index=False)
    )
    print(f"\nBest strategy by Sharpe: {best_strategy}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [2, 1, 1]})
    equity_df.plot(ax=axes[0], linewidth=2)
    axes[0].set_title("Refined A2C + PPO Ensemble: OOS Equity Curves")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].grid(alpha=0.3)

    plot_summary = summary.set_index("strategy")
    plot_summary["sharpe"].sort_values().plot(kind="barh", ax=axes[1], color="#357ABD")
    axes[1].set_title("Sharpe by Strategy")
    axes[1].grid(axis="x", alpha=0.3)

    (plot_summary["mdd"].sort_values() * 100).plot(kind="barh", ax=axes[2], color="#C44E52")
    axes[2].set_title("MDD by Strategy (%)")
    axes[2].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "refined_strategy_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close("all")

    dynamic_cols = [
        "Auto_DD_Guard_v1",
        "Auto_DD_Guard_v2_Strong",
        "Regime_DD_Guard",
        "Turnover_Aware_DD_Guard",
        "Vol_Target_DD_Guard",
        "Regime_Tactical",
    ]
    plt.figure(figsize=(14, 6))
    for col in dynamic_cols:
        ratio_df[col].plot(label=col, linewidth=1.8)
    plt.title("Refined Dynamic Strategies: PPO Ratio")
    plt.ylabel("PPO ratio")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "refined_dynamic_ppo_ratio.png"), dpi=160, bbox_inches="tight")
    plt.close("all")

    print("\nSaved files:")
    print(f"- {RESULT_DIR}/refined_summary.csv")
    print(f"- {RESULT_DIR}/refined_fold_results.csv")
    print(f"- {RESULT_DIR}/refined_strategy_comparison.png")
    print(f"- {RESULT_DIR}/refined_dynamic_ppo_ratio.png")


if __name__ == "__main__":
    main()
