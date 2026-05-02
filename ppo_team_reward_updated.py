"""
PPO baseline code for Ensemble Team
===================================
Purpose
- This file provides a PPO baseline that can be used as the common environment for PPO + A2C ensemble.
- It mixes the good parts of two existing notebooks:
  1) Portfolio weight allocation structure from the original HMM + PPO code
  2) Technical indicators + Turbulence idea from PPO_Trading_v2_with_TA_State

Bug fixes applied (v2)
- [REWARD UPDATE] Applied Reward Rescaling + SPY Excess Return + explicit Turnover penalty:
    reward += 0.002 * clipped_rolling_sharpe
    reward += 0.3 * clipped(port_ret - spy_ret)
    reward -= 0.002 * turnover

- [FIX 1] Look-ahead bias in PortfolioEnv.step():
    Before: obs[t] → action → returns[t]   (today's signal → today's return)
    After:  obs[t] → action → returns[t+1] (today's signal → tomorrow's return)
    The state at time t is built from today's closing prices/indicators.
    The realistic return is the one realized on the *next* day (t+1).
    Implementation: self.returns has shape (T,). In step(), we consume
    self.returns[self.t + 1] before incrementing self.t, and we terminate
    one step earlier so we never read out of bounds.

- [FIX 2] HMM and StandardScaler fitted on full data (look-ahead in walk-forward):
    Before: build_dataset() fitted scaler + HMM on ALL rows, then passed the
            pre-processed DataFrame to run_walk_forward(). The test-period
            distribution was therefore visible to the HMM/scaler during fitting.
    After:  build_dataset() now returns only *raw* features (no scaler, no HMM).
            run_walk_forward() calls fit_fold_features() at the start of each
            fold, which fits scaler + HMM exclusively on that fold's training
            rows, then transforms both train and test rows. This makes the
            walk-forward truly leakage-free.

Recommended ensemble usage
- Keep this data/feature/env/reward/walk-forward structure fixed.
- In the ensemble code, add A2C by changing only the model constructor.
- PPO and A2C should use the same PortfolioEnv, state, reward, train/test split.

Required packages
pip install yfinance hmmlearn stable-baselines3[extra] gymnasium scikit-learn matplotlib pandas numpy requests
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================
# 0. Global settings
# ============================================================
FRED_API_KEY = "PUT_YOUR_FRED_API_KEY_HERE"   # 팀원이 본인 FRED API key로 교체
DOWNLOAD_START = "2005-01-01"
START_DATE = "2006-01-01"
END_DATE = "2026-04-02"
RANDOM_SEED = 42

ASSETS = ["SPY", "TLT", "SHV", "GLD", "DBC"]
N_ASSETS = len(ASSETS)
N_STATES = 6

TRAIN_YEARS = 4
TEST_YEARS = 1
TIMESTEPS = 50_000        # 빠른 실험용. 최종 실험은 100_000~300_000 가능
TC = 0.001                # 거래비용

# Reward design coefficients
# - Rolling Sharpe 계수는 0.01에서 0.002로 낮춰 안정성 보너스가 수익률을 압도하지 않게 함
# - Excess Return 계수는 0.3으로 설정해 상승장에서 SPY 대비 소외되는 행동을 줄임
# - Turnover 페널티를 보상에도 직접 반영해 과도한 리밸런싱을 억제함
ROLLING_SHARPE_COEF = 0.002
EXCESS_RETURN_COEF = 0.3
TURNOVER_REWARD_COEF = 0.002

HMM_FEATURES = [
    "Equity_vs_Bond",
    "US10Y",
    "Jobless_Claims",
    "HY_Spread",
    "VIX",
    "DXY",
    "Copper_Gold_Ratio",
    "Market_Breadth",
]


# ============================================================
# 1. Utility functions
# ============================================================
def fetch_fred(series_id: str, api_key: str, start: str = DOWNLOAD_START, end: str = END_DATE) -> pd.Series:
    """Fetch one FRED series. If failed, return empty series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        s = pd.Series(
            {pd.to_datetime(o["date"]): np.nan if o["value"] == "." else float(o["value"]) for o in obs},
            name=series_id,
        )
        return s.sort_index()
    except Exception as e:
        print(f"[WARN] FRED {series_id} fetch failed: {e}")
        return pd.Series(dtype=float, name=series_id)


def calc_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def clip_1(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).clip(-1, 1)


def compute_turbulence_index(price_df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Turbulence Index: measures how unusual today's asset return vector is
    compared with the past lookback-day distribution.
    """
    returns = price_df.pct_change().dropna()
    turbulence = pd.Series(index=returns.index, dtype=float, name="Turbulence")

    for i in range(lookback, len(returns)):
        hist = returns.iloc[i - lookback:i]
        curr = returns.iloc[i]
        mu = hist.mean().values
        cov = hist.cov().values
        try:
            cov_inv = np.linalg.pinv(cov)
            diff = (curr.values - mu).reshape(1, -1)
            turbulence.iloc[i] = float(diff @ cov_inv @ diff.T)
        except Exception:
            turbulence.iloc[i] = np.nan

    return turbulence


# ============================================================
# 2. Data loading — returns RAW (unscaled, no HMM) DataFrame
# ============================================================
def build_dataset() -> pd.DataFrame:
    """
    Download and engineer raw features.

    Returns
    -------
    df_raw : DataFrame
        Contains asset prices, raw (unscaled) HMM feature columns,
        asset daily returns, and technical indicators.
        HMM_FEATURES columns are NOT yet standardized — scaling and
        HMM fitting happen per fold inside run_walk_forward().

    NOTE (FIX 2)
    ------------
    The original code called StandardScaler.fit_transform() and
    GaussianHMM.fit() on the entire date range here, leaking future
    distribution information into every training fold.  That code has
    been removed.  Scaler/HMM fitting now happens inside
    fit_fold_features(), called once per fold using only train rows.
    """
    print("[1/4] Downloading ETF and macro data...")

    yf_tickers = sorted(set(ASSETS + ["SPY", "TLT", "RSP", "EEM"]))
    df_raw = yf.download(yf_tickers, start=DOWNLOAD_START, end=END_DATE, auto_adjust=True, progress=False)
    df_price = df_raw["Close"].copy() if isinstance(df_raw.columns, pd.MultiIndex) else df_raw.copy()
    df_price.index = pd.to_datetime(df_price.index).tz_localize(None).normalize()
    df_price = df_price.ffill().bfill()

    macro_tickers = {
        "^VIX": "VIX",
        "DX-Y.NYB": "DXY",
        "HG=F": "Copper",
        "GC=F": "Gold",
    }
    df_macro_raw = yf.download(list(macro_tickers.keys()), start=DOWNLOAD_START, end=END_DATE, auto_adjust=True, progress=False)
    df_macro = df_macro_raw["Close"].copy() if isinstance(df_macro_raw.columns, pd.MultiIndex) else df_macro_raw.copy()
    df_macro = df_macro.rename(columns=macro_tickers)
    df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None).normalize()
    df_macro = df_macro.ffill().bfill()

    print("[2/4] Downloading FRED data...")
    fred_map = {
        "DGS10": "US10Y",
        "ICSA": "Jobless_Claims",
        "BAMLH0A0HYM2": "HY_Spread",
    }
    fred_series = []
    for sid, name in fred_map.items():
        s = fetch_fred(sid, FRED_API_KEY).rename(name)
        fred_series.append(s)
    df_fred = pd.concat(fred_series, axis=1) if fred_series else pd.DataFrame()
    if not df_fred.empty:
        df_fred.index = pd.to_datetime(df_fred.index).tz_localize(None).normalize()
        df_fred = df_fred.ffill().bfill()

    print("[3/4] Building raw HMM features and technical indicators...")
    df = df_price.join(df_macro, how="left").join(df_fred, how="left")
    df = df.ffill().bfill()

    # --- Raw (unscaled) HMM features ---
    df["Equity_vs_Bond"] = df["SPY"] / (df["TLT"] + 1e-8)
    df["Copper_Gold_Ratio"] = df["Copper"] / (df["Gold"] + 1e-8)
    df["Market_Breadth"] = (df["SPY"] / (df["RSP"] + 1e-8)).pct_change(21)

    df = df.dropna(subset=HMM_FEATURES).copy()
    df = df[df.index >= pd.to_datetime(START_DATE)]

    # --- Asset returns (t→t+1 shift applied in the environment, not here) ---
    ret_cols = []
    for ticker in ASSETS:
        df[f"{ticker}_ret"] = df[ticker].pct_change().fillna(0)
        ret_cols.append(f"{ticker}_ret")

    # --- Technical indicators (these use only past prices; no scaling needed) ---
    technical_cols = []
    for ticker in ASSETS:
        price = df[ticker]
        ret = price.pct_change()

        df[f"{ticker}_mom21"] = clip_1(price.pct_change(21))
        df[f"{ticker}_vol60"] = (ret.rolling(60).std() * np.sqrt(252) / 0.80).clip(0, 1)

        rsi = calc_rsi(price, window=14)
        df[f"{ticker}_RSI14"] = clip_1((rsi - 50) / 50)

        ma60 = price.rolling(60).mean()
        df[f"{ticker}_MA60_DIV"] = clip_1((price / (ma60 + 1e-8) - 1) / 0.20)

        ma20 = price.rolling(20).mean()
        std20 = price.rolling(20).std()
        df[f"{ticker}_BB20_POS"] = clip_1((price - ma20) / (2 * std20 + 1e-8))

        technical_cols += [
            f"{ticker}_mom21",
            f"{ticker}_vol60",
            f"{ticker}_RSI14",
            f"{ticker}_MA60_DIV",
            f"{ticker}_BB20_POS",
        ]

    df["SPY_Mom_3M"] = clip_1(df["SPY"].pct_change(63))

    # Turbulence: uses only past 252-day window internally, so it is already
    # look-ahead-free. We keep it in the raw dataset.
    turbulence = compute_turbulence_index(df[ASSETS], lookback=252)
    df["Turbulence"] = turbulence.reindex(df.index).ffill().bfill().fillna(0)

    keep_cols = ASSETS + ret_cols + HMM_FEATURES + ["SPY_Mom_3M"] + technical_cols + ["Turbulence"]
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    print("[4/4] Raw dataset ready")
    print(f"    Rows : {len(df):,}")
    print(f"    Cols : {df.shape[1]}")
    print(f"    Range: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


# ============================================================
# 2b. Per-fold feature engineering (FIX 2)
# ============================================================
def fit_fold_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """
    Fit StandardScaler + GaussianHMM on TRAIN rows only, then transform
    both train and test rows.  Returns augmented DataFrames and the
    list of state columns to feed the RL environment.

    This function replaces the single global scaler/HMM call that used to
    live in build_dataset(), eliminating the look-ahead bias described in FIX 2.

    Parameters
    ----------
    df_train, df_test : DataFrames with raw (unscaled) HMM_FEATURES columns.

    Returns
    -------
    df_tr_out, df_te_out : DataFrames with scaled HMM features + HMM regime columns appended.
    state_cols           : list of column names that form the RL observation vector.
    turbulence_threshold : 99th-percentile turbulence computed on TRAIN only.
    """
    regime_prob_cols = [f"Regime_prob_{i}" for i in range(N_STATES)]

    def _process(df_in: pd.DataFrame, scaler: StandardScaler, hmm: GaussianHMM) -> pd.DataFrame:
        df = df_in.copy()
        raw = df[HMM_FEATURES].replace([np.inf, -np.inf], np.nan).ffill().bfill().values
        scaled = scaler.transform(raw)
        df[HMM_FEATURES] = scaled                        # overwrite with scaled values
        probs = hmm.predict_proba(scaled)
        df[regime_prob_cols] = probs
        df["HMM_Regime"] = hmm.predict(scaled)
        return df

    # Fit on train only
    scaler = StandardScaler()
    raw_train = df_train[HMM_FEATURES].replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    scaled_train = scaler.fit_transform(raw_train)

    hmm = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=RANDOM_SEED,
    )
    hmm.fit(scaled_train)

    # Turbulence threshold from train period only
    turb_threshold = float(df_train["Turbulence"].quantile(0.99))

    # Transform both splits
    df_tr_out = _process(df_train, scaler, hmm)
    df_te_out = _process(df_test, scaler, hmm)

    # Scale turbulence for state input (threshold computed on train)
    for df_out in (df_tr_out, df_te_out):
        df_out["Turbulence_scaled"] = (
            df_out["Turbulence"] / (turb_threshold + 1e-8)
        ).clip(0, 3) / 3

    technical_cols = []
    for ticker in ASSETS:
        technical_cols += [
            f"{ticker}_mom21",
            f"{ticker}_vol60",
            f"{ticker}_RSI14",
            f"{ticker}_MA60_DIV",
            f"{ticker}_BB20_POS",
        ]

    state_cols = (
        HMM_FEATURES
        + ["SPY_Mom_3M"]
        + regime_prob_cols
        + technical_cols
        + ["Turbulence_scaled"]
    )

    return df_tr_out, df_te_out, state_cols, turb_threshold


# ============================================================
# 3. Portfolio weight allocation environment
# ============================================================
class PortfolioEnv(gym.Env):
    """
    Weight-allocation environment.

    FIX 1 — Look-ahead bias correction
    ------------------------------------
    Old behaviour (biased):
        _obs() returns features[t]
        step() consumes returns[t]   ← same timestamp → today's signal → today's return
    New behaviour (correct):
        _obs() returns features[t]
        step() consumes returns[t+1] ← next day's return
        Episode length is T-1 steps (last state has no next-day return).

    Concretely: at step t the agent sees the state built from day-t closing
    prices, decides on portfolio weights, and receives the return that accrues
    between day-t close and day-(t+1) close.  This matches real execution:
    you observe prices at today's close, place orders, and profit/loss is
    measured at tomorrow's close.

    Important:
    - Action is ETF portfolio weight, not buy/sell quantity.
    - This is better for ETF allocation and easier to share with A2C/SAC ensemble.
    - A2C can use this same environment without any modification.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df_period: pd.DataFrame,
        state_cols: list[str],
        assets: list[str] = ASSETS,
        turbulence_threshold: float | None = None,
        tc: float = TC,
    ):
        super().__init__()
        self.df = df_period.reset_index(drop=True).copy()
        self.state_cols = state_cols
        self.assets = assets
        self.ret_cols = [f"{a}_ret" for a in assets]
        self.n_assets = len(assets)
        self.tc = tc
        self.turbulence_threshold = turbulence_threshold

        self.features = self.df[self.state_cols].astype(np.float32).values
        self.returns = self.df[self.ret_cols].astype(np.float32).values
        self.spy_returns = self.df["SPY_ret"].astype(np.float32).values
        self.turbulence = self.df["Turbulence"].astype(np.float32).values

        # FIX 1: episode has T-1 steps because we read returns[t+1]
        self.max_t = len(self.df) - 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(state_cols),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.reset()

    def _obs(self) -> np.ndarray:
        return self.features[self.t].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.prev_w = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.recent_returns = []
        return self._obs(), {}

    def step(self, action):
        # Convert raw action to valid portfolio weights
        w = np.clip(action, 1e-8, 1.0).astype(np.float32)
        w = w / (w.sum() + 1e-8)

        # FIX 1: use next-day return (t+1) instead of same-day return (t)
        # We consume returns[self.t + 1] while still observing state[self.t].
        next_t = self.t + 1
        raw_port_ret = float(np.dot(w, self.returns[next_t]))
        turnover = float(np.sum(np.abs(w - self.prev_w)))
        port_ret = raw_port_ret - self.tc * turnover
        port_ret_safe = float(np.clip(port_ret, -0.99, 0.99))

        # Portfolio value update
        self.portfolio_value *= (1.0 + port_ret_safe)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_dd = 1.0 - self.portfolio_value / (self.peak_value + 1e-8)

        # Reward = return + rescaled stability + SPY excess + drawdown penalty + turnover penalty + turbulence penalty
        reward = float(np.log1p(port_ret_safe))

        self.recent_returns.append(port_ret_safe)
        if len(self.recent_returns) > 20:
            self.recent_returns.pop(0)

        if len(self.recent_returns) >= 5:
            rr = np.array(self.recent_returns, dtype=np.float32)
            rolling_sharpe = float(rr.mean() / (rr.std() + 1e-8) * np.sqrt(252))
        else:
            rolling_sharpe = 0.0
        # Step 1. Reward Rescaling: prevent rolling Sharpe from dominating daily return
        reward += ROLLING_SHARPE_COEF * np.clip(rolling_sharpe, -3, 3)

        # FIX 1: SPY benchmark return also uses t+1 to stay consistent
        spy_ret = float(self.spy_returns[next_t])
        excess = float(np.clip(port_ret_safe - spy_ret, -0.05, 0.05))
        # Step 2. Benchmark Excess Return: reduce FOMO/missed-upside behavior vs SPY
        reward += EXCESS_RETURN_COEF * excess

        reward -= 0.05 * max(0.0, current_dd - 0.10)

        # Explicit turnover penalty: TC already reduces return, this additionally discourages over-trading
        reward -= TURNOVER_REWARD_COEF * turnover

        if self.turbulence_threshold is not None:
            if float(self.turbulence[self.t]) > self.turbulence_threshold:
                # Soft risk penalty, not forced liquidation.
                reward -= 0.01

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": port_ret_safe,
            "raw_portfolio_return": raw_port_ret,
            "turnover": turnover,
            "drawdown": current_dd,
            "weights": w.copy(),
            "rolling_sharpe": rolling_sharpe,
            "excess_return_vs_spy": excess,
            "reward_rolling_sharpe_coef": ROLLING_SHARPE_COEF,
            "reward_excess_coef": EXCESS_RETURN_COEF,
            "reward_turnover_coef": TURNOVER_REWARD_COEF,
            "turbulence": float(self.turbulence[self.t]),
        }

        self.prev_w = w
        self.t += 1
        # FIX 1: terminate when there is no t+1 return left to consume
        terminated = self.t >= self.max_t
        return self._obs(), float(reward), terminated, False, info


# ============================================================
# 4. PPO model maker
# ============================================================
def make_ppo(env, prev_model=None):
    """
    PPO baseline.
    For the ensemble team, A2C can be added by using the same env and replacing only this function.
    """
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        seed=RANDOM_SEED,
        verbose=0,
    )

    # Fine-tuning: initialize current fold with previous fold parameters
    if prev_model is not None:
        try:
            model.set_parameters(prev_model.get_parameters())
        except Exception as e:
            print(f"[WARN] Failed to copy previous PPO parameters: {e}")

    return model


# ============================================================
# 5. Backtest helpers
# ============================================================
def compute_metrics(values: pd.Series, freq: int = 252) -> dict:
    values = pd.Series(values).dropna()
    rets = values.pct_change().dropna()
    if len(values) < 2 or len(rets) < 2:
        return {"total_return": np.nan, "annual_return": np.nan, "volatility": np.nan, "sharpe": np.nan, "mdd": np.nan}

    total_return = values.iloc[-1] / values.iloc[0] - 1
    annual_return = (values.iloc[-1] / values.iloc[0]) ** (freq / len(rets)) - 1
    volatility = rets.std() * np.sqrt(freq)
    sharpe = annual_return / (volatility + 1e-8)
    mdd = (values / values.cummax() - 1).min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "mdd": mdd,
    }


def run_policy(model, df_test: pd.DataFrame, state_cols: list[str], turbulence_threshold: float) -> pd.DataFrame:
    env = PortfolioEnv(df_test, state_cols, turbulence_threshold=turbulence_threshold)
    obs, _ = env.reset()

    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append(info)

    out = pd.DataFrame(records)
    out["portfolio_value"] = out["portfolio_value"].astype(float)
    return out


# ============================================================
# 6. Walk-forward PPO baseline
# ============================================================
def run_walk_forward(df_raw: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Walk-forward backtest.

    FIX 2 applied here: for each fold we call fit_fold_features(), which
    fits the StandardScaler and GaussianHMM exclusively on df_train rows
    and then transforms both df_train and df_test.  The test period's
    distribution is therefore never seen during fitting.

    Parameters
    ----------
    df_raw : raw DataFrame returned by build_dataset() (no scaler/HMM applied).
    """
    print("\nRunning PPO walk-forward backtest...")
    print(f"Train: {TRAIN_YEARS} years / Test: {TEST_YEARS} year")

    start = df_raw.index.min()
    end = df_raw.index.max()
    train_start = start
    train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)

    all_values = []
    all_infos = []
    prev_model = None
    period = 1
    global_value = 1.0

    while train_end + pd.DateOffset(years=TEST_YEARS) <= end:
        test_end = train_end + pd.DateOffset(years=TEST_YEARS)

        df_train_raw = df_raw[(df_raw.index >= train_start) & (df_raw.index < train_end)].copy()
        df_test_raw  = df_raw[(df_raw.index >= train_end)   & (df_raw.index < test_end)].copy()

        if len(df_train_raw) < 252 * 2 or len(df_test_raw) < 50:
            train_start += pd.DateOffset(years=1)
            train_end   += pd.DateOffset(years=1)
            continue

        print(f"\n[Period {period}] Train: {train_start.date()}~{train_end.date()} "
              f"| Test: {train_end.date()}~{test_end.date()}")

        # FIX 2: fit scaler + HMM on train rows only, transform both splits
        df_train, df_test, state_cols, turb_threshold = fit_fold_features(
            df_train_raw, df_test_raw
        )

        train_env = DummyVecEnv([
            lambda: Monitor(PortfolioEnv(df_train, state_cols, turbulence_threshold=turb_threshold))
        ])

        model = make_ppo(train_env, prev_model=prev_model)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
        prev_model = model

        test_result = run_policy(model, df_test, state_cols, turb_threshold)

        # Convert period-local PV to global continuous PV
        period_values = test_result["portfolio_value"].values
        period_values = global_value * period_values / period_values[0]
        global_value = float(period_values[-1])

        pv_series = pd.Series(period_values, index=df_test.index[:len(period_values)], name="PPO")
        all_values.append(pv_series)

        test_info = test_result.copy()
        test_info.index = df_test.index[:len(test_info)]
        test_info["period"] = period
        all_infos.append(test_info)

        metrics = compute_metrics(pv_series)
        print(
            f"  Total:{metrics['total_return']:+.2%} | "
            f"Ann:{metrics['annual_return']:+.2%} | "
            f"Vol:{metrics['volatility']:.2%} | "
            f"Sharpe:{metrics['sharpe']:.3f} | "
            f"MDD:{metrics['mdd']:.2%}"
        )

        # Rolling-window update
        train_start += pd.DateOffset(years=1)
        train_end   += pd.DateOffset(years=1)
        period += 1

    if not all_values:
        raise RuntimeError("No valid walk-forward periods were created.")

    pv_all   = pd.concat(all_values).sort_index()
    info_all = pd.concat(all_infos).sort_index()
    return pv_all, info_all


# ============================================================
# 7. Main execution
# ============================================================
if __name__ == "__main__":
    df_raw = build_dataset()
    ppo_pv, ppo_info = run_walk_forward(df_raw)

    metrics = compute_metrics(ppo_pv)
    print("\n========== PPO Baseline Final Metrics ==========")
    for k, v in metrics.items():
        if k in ["total_return", "annual_return", "volatility", "mdd"]:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.3f}")

    ppo_pv.to_csv("ppo_baseline_portfolio_value.csv", encoding="utf-8-sig")
    ppo_info.to_csv("ppo_baseline_trade_info.csv", encoding="utf-8-sig")
    print("\nSaved:")
    print("- ppo_baseline_portfolio_value.csv")
    print("- ppo_baseline_trade_info.csv")
