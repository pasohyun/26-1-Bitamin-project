#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2C + PPO 앙상블 비율 실험 노트북 빌더
실행: python build_ensemble_compare_nb.py
결과:
  - A2C_PPO_Ensemble_Experiment.ipynb
  - Ensemble_Comparison.ipynb
"""

import json

OUTPUT_NB = "A2C_PPO_Ensemble_Experiment.ipynb"

cells = []


def src(text):
    lines = text.expandtabs(4).split("\n")
    out = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + "\n")
        elif line:
            out.append(line)
    return out


def md(text, cell_id=None):
    cells.append(
        {
            "cell_type": "markdown",
            "id": cell_id or f"m{len(cells)}",
            "metadata": {},
            "source": src(text),
        }
    )


def code(text, cell_id=None):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": cell_id or f"c{len(cells)}",
            "metadata": {},
            "outputs": [],
            "source": src(text),
        }
    )


md(
    """# A2C + PPO 앙상블 비율 실험

담당 범위: **앙상블 모델(A2C + PPO) - 앙상블 비율, 자동/수동 조절 실험**

## 공통 조건

```python
FEATURES = ['Equity_vs_Bond','US10Y','Jobless_Claims_MA','VIX',
            'Market_Breadth','spread_10y2y','DXY','Copper_Gold','SPY_Mom_3M']
OBSERVATION = 9개 매크로 피처 + HMM 6개 국면확률 = 15차원

REWARD = log(1+r) + 0.01 * RollingSharpe - 0.05 * max(0, DD - 0.10)

WALK_FORWARD = 4년 Train / 1년 Test / 일별 리밸런싱 백테스트
TIMESTEPS = 300_000
NET_ARCH = [256, 256]
ASSETS = ['SPY','TLT','SHV','GLD','DBC']
```

## 실험 전략

| 구분 | 전략 | 의미 |
|---|---|---|
| 단일 모델 | PPO Only | PPO 100% |
| 단일 모델 | A2C Only | A2C 100% |
| 수동 비율 | Manual 25/50/75 | 사람이 정한 PPO:A2C 고정 비율 |
| 자동 조절 | Auto Rolling Sharpe | 최근 성과가 좋은 모델 비중 자동 확대 |
| 자동 방어 | Auto DD Guard | Auto Rolling Sharpe + DD 10% 초과 시 PPO 비중 제한 |
| 수동 개입 | Manual VIX Override | 평소 50:50, VIX 위험 구간에서 A2C 100% |

백테스팅은 각 walk-forward fold의 **1년 OOS Test 구간**에서 수행합니다. 즉 학습에 쓰지 않은 테스트 기간에 PPO/A2C가 낸 비중을 앙상블하고, 일별 리밸런싱 포트폴리오 수익률을 누적해 Sharpe, MDD, Calmar, turnover를 비교합니다.

주의: HMM 6개 국면확률을 observation에 추가하므로, HMM 없는 9차원 입력으로 학습된 기존 `ensemble_models/` 모델은 재사용할 수 없습니다. 이 노트북은 PPO/A2C를 다시 학습합니다.

노트북은 기본적으로 일별 리밸런싱(`B`)입니다. 월간 단위 실험은 `REBALANCE_FREQ = 'ME'`, `ANNUALIZATION = 12`, `SHARPE_WINDOW = 12`로 바꾸면 됩니다."""
)


code(
    """import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    # 이 conda 환경에서는 PyTorch DLL 로딩 안정성을 위해 stable_baselines3를 먼저 import합니다.
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "필수 패키지가 없습니다. 터미널에서 `pip install stable-baselines3 gymnasium` 실행 후 다시 실행하세요."
    ) from e

import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)

CACHE_PATH = "cached_raw_data.csv"
RESULT_DIR = "ensemble_results"
MODEL_DIR = "ensemble_models"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Equity_vs_Bond", "US10Y", "Jobless_Claims_MA", "VIX",
    "Market_Breadth", "spread_10y2y", "DXY", "Copper_Gold", "SPY_Mom_3M",
]
ASSETS = ["SPY", "TLT", "SHV", "GLD", "DBC"]
HMM_N_STATES = 6
HMM_PROB_COLS = [f"HMM_Prob_{i}" for i in range(HMM_N_STATES)]
OBS_COLS = FEATURES + HMM_PROB_COLS

REBALANCE_FREQ = "B"        # 일별: B, 월별: ME, 분기별: QE
ANNUALIZATION = 252         # 일별 252, 월별 12, 분기별 4
SHARPE_WINDOW = 252         # 일별 252스텝 = 1년
TRAIN_YEARS = 4
TEST_YEARS = 1

TIMESTEPS = 300_000
NET_ARCH = [256, 256]
TRANSACTION_COST = 0.0005   # turnover 1.0당 5bp 가정

FAST_DEV_RUN = False        # True면 빠른 코드 점검용: 1개 fold, 10_000 timesteps
SAVE_MODELS = True

if FAST_DEV_RUN:
    TIMESTEPS = 10_000

print("설정 완료")
print(f"리밸런싱={REBALANCE_FREQ}, timesteps={TIMESTEPS:,}, assets={ASSETS}")
print(f"관측 차원={len(OBS_COLS)} = 피처 {len(FEATURES)}개 + HMM 국면확률 {HMM_N_STATES}개")"""
)


md(
    """## 데이터 준비

`cached_raw_data.csv`에서 가격/매크로 데이터를 읽고, 기존 프로젝트와 같은 피처 엔지니어링 방식을 사용합니다.

주의: 관측 피처와 HMM 국면확률은 `shift(1)`로 한 기간 lag를 줍니다. 즉 이번 월말 수익률을 예측할 때 이전 월말까지 관측 가능한 정보만 사용합니다."""
)


code(
    """raw = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True).sort_index()

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

X_all = dataset[[f"x_{c}" for c in FEATURES]].copy()
X_all.columns = FEATURES

raw_signal_all = dataset[[f"raw_{c}" for c in ["VIX", "spread_10y2y", "SPY_Mom_3M"]]].copy()
raw_signal_all.columns = ["VIX", "spread_10y2y", "SPY_Mom_3M"]

R_all = dataset[[f"r_{c}" for c in ASSETS]].copy()
R_all.columns = ASSETS

print("데이터 준비 완료")
print(f"기간: {dataset.index.min().date()} ~ {dataset.index.max().date()}")
print(f"샘플 수: {len(dataset)} / 피처: {X_all.shape[1]} / 자산: {R_all.shape[1]}")
display(pd.concat([X_all.head(3), R_all.head(3)], axis=1))"""
)


md(
    """## 공통 보상 함수 환경

두 모델(PPO, A2C)이 같은 환경과 같은 보상 함수를 쓰도록 통일합니다.

```python
reward = (
    np.log(1 + port_ret)
    + 0.01 * rolling_sharpe
    - 0.05 * max(0, current_dd - 0.10)
)
```

환경 내부 상태:

- `return_buffer = deque(maxlen=252)`
- `port_val = 1.0`
- `peak_val = 1.0`

액션은 실수 벡터로 받고 softmax로 변환해 long-only, fully-invested 포트폴리오 비중으로 사용합니다."""
)


code(
    """def softmax_action(action):
    z = np.asarray(action, dtype=np.float64).reshape(-1)
    z = np.clip(z, -20, 20)
    exp_z = np.exp(z - np.max(z))
    return exp_z / (exp_z.sum() + 1e-12)


class PortfolioAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features,
        returns,
        transaction_cost=TRANSACTION_COST,
        sharpe_window=SHARPE_WINDOW,
        annualization=ANNUALIZATION,
    ):
        super().__init__()
        self.features_df = features.copy()
        self.returns_df = returns.copy()
        self.features = self.features_df.values.astype(np.float32)
        self.returns = self.returns_df.values.astype(np.float32)
        self.transaction_cost = transaction_cost
        self.sharpe_window = sharpe_window
        self.annualization = annualization
        self.n_assets = self.returns.shape[1]

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.features.shape[1],),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.return_buffer = deque(maxlen=self.sharpe_window)
        self.port_val = 1.0
        self.peak_val = 1.0
        self.prev_w = np.ones(self.n_assets) / self.n_assets
        return self.features[self.idx], {}

    def step(self, action):
        w = softmax_action(action)
        r = self.returns[self.idx]

        gross_ret = float(np.dot(w, r))
        turnover = float(np.sum(np.abs(w - self.prev_w)) / 2.0)
        cost = turnover * self.transaction_cost
        port_ret = max(gross_ret - cost, -0.999999)

        base = float(np.log(1.0 + port_ret))

        self.return_buffer.append(port_ret)
        if len(self.return_buffer) >= 3:
            buf = np.array(self.return_buffer, dtype=np.float64)
            rolling_sharpe = float(buf.mean() / (buf.std() + 1e-8) * np.sqrt(self.annualization))
        else:
            rolling_sharpe = 0.0

        self.port_val *= 1.0 + port_ret
        self.peak_val = max(self.peak_val, self.port_val)
        current_dd = float((self.peak_val - self.port_val) / (self.peak_val + 1e-8))

        reward = base + 0.01 * rolling_sharpe - 0.05 * max(0.0, current_dd - 0.10)

        self.prev_w = w
        self.idx += 1
        terminated = self.idx >= len(self.returns)
        truncated = False
        obs = self.features[-1] if terminated else self.features[self.idx]

        info = {
            "weights": w,
            "gross_ret": gross_ret,
            "port_ret": port_ret,
            "turnover": turnover,
            "cost": cost,
            "rolling_sharpe": rolling_sharpe,
            "drawdown": current_dd,
            "portfolio_value": self.port_val,
        }
        return obs, float(reward), terminated, truncated, info


print("환경 정의 완료")"""
)


md(
    """## Walk-forward split

4년 학습, 1년 테스트를 rolling 방식으로 생성합니다. 데이터 끝까지 가능한 fold만 자동으로 만듭니다."""
)


code(
    """def make_walk_forward_splits(index, train_years=TRAIN_YEARS, test_years=TEST_YEARS):
    first_year = int(index.min().year)
    last_date = index.max()
    splits = []

    start_year = first_year
    while True:
        train_start = pd.Timestamp(f"{start_year}-01-01")
        train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)

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

        start_year += test_years
        if test_end >= last_date:
            break

    return splits


splits = make_walk_forward_splits(dataset.index)
if FAST_DEV_RUN:
    splits = splits[-1:]

split_table = pd.DataFrame(
    [
        {
            "fold": s["fold"],
            "train": f"{s['train_start'].date()} ~ {s['train_end'].date()}",
            "test": f"{s['test_start'].date()} ~ {s['test_end'].date()}",
            "n_train": int(s["train_mask"].sum()),
            "n_test": int(s["test_mask"].sum()),
        }
        for s in splits
    ]
)
display(split_table)"""
)


md(
    """## HMM 6개 국면확률 생성

각 walk-forward fold마다 HMM은 **Train 구간의 daily macro feature**로만 fit합니다. 이후 train/test daily feature에 대해 6개 국면확률을 계산하고, 월말 또는 분기말 기준으로 마지막 값을 가져온 뒤 `shift(1)`을 적용해 수익률 예측 시점보다 한 기간 앞선 정보만 observation에 붙입니다.

국면 번호는 HMM label switching을 줄이기 위해 train 구간의 VIX 평균이 낮은 순서로 재정렬합니다. 따라서 `HMM_Prob_0`은 상대적으로 저변동 국면, `HMM_Prob_5`는 상대적으로 고변동 국면에 가깝습니다."""
)


code(
    """def make_fold_hmm_prob_features(split, feature_daily):
    train_daily = feature_daily.loc[
        (feature_daily.index >= split["train_start"]) &
        (feature_daily.index <= split["train_end"])
    ].copy()
    inference_daily = feature_daily.loc[
        (feature_daily.index >= split["train_start"]) &
        (feature_daily.index <= split["test_end"])
    ].copy()

    if len(train_daily) < HMM_N_STATES * 20:
        raise ValueError(f"HMM 학습 데이터가 너무 적습니다: {len(train_daily)} rows")

    hmm_scaler = StandardScaler()
    train_scaled = hmm_scaler.fit_transform(train_daily[FEATURES])
    inference_scaled = hmm_scaler.transform(inference_daily[FEATURES])

    hmm = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type="full",
        n_iter=1000,
        random_state=SEED + split["fold"],
    )
    hmm.fit(train_scaled)

    train_states = hmm.predict(train_scaled)
    vix_by_state = pd.Series(train_daily["VIX"].values).groupby(train_states).mean()
    ordered_states = vix_by_state.sort_values().index.tolist()

    raw_probs = hmm.predict_proba(inference_scaled)
    ranked_probs = np.zeros_like(raw_probs)
    for new_rank, old_state in enumerate(ordered_states):
        ranked_probs[:, new_rank] = raw_probs[:, old_state]

    daily_probs = pd.DataFrame(ranked_probs, index=inference_daily.index, columns=HMM_PROB_COLS)
    period_probs = daily_probs.resample(REBALANCE_FREQ).last().shift(1)
    return period_probs


def append_hmm_probs(X_train, X_test, R_train, R_test, split):
    period_probs = make_fold_hmm_prob_features(split, feature_daily)
    train_probs = period_probs.reindex(X_train.index)
    test_probs = period_probs.reindex(X_test.index)

    X_train_obs = pd.concat([X_train, train_probs], axis=1).dropna()
    X_test_obs = pd.concat([X_test, test_probs], axis=1).dropna()

    R_train_obs = R_train.loc[X_train_obs.index]
    R_test_obs = R_test.loc[X_test_obs.index]

    return X_train_obs[OBS_COLS], X_test_obs[OBS_COLS], R_train_obs, R_test_obs


print("HMM 국면확률 함수 준비 완료")"""
)


md(
    """## 백테스팅 설계

백테스트는 별도 파일로 분리하지 않고, 아래 전체 실험 셀 안에서 walk-forward 방식으로 함께 실행합니다.

1. 각 fold에서 4년 Train 데이터로 PPO/A2C를 각각 학습합니다.
2. 같은 fold의 1년 Test 데이터는 학습에 쓰지 않고 OOS 백테스트 구간으로 둡니다.
3. Test 월마다 두 모델의 자산 비중을 예측합니다.
4. 수동/자동 앙상블 비율로 최종 비중을 만들고, 거래비용을 차감한 포트폴리오 수익률을 누적합니다.
5. 모든 fold의 OOS 수익률을 이어 붙여 최종 Sharpe, MDD, Calmar, CAGR, 평균 turnover를 비교합니다.

따라서 이 노트북의 핵심 비교 기준은 학습 reward가 아니라 **학습 이후 OOS 백테스트 성과**입니다."""
)


md(
    """## PPO/A2C 학습 및 예측 함수

두 모델은 같은 `NET_ARCH = [256, 256]`, 같은 환경, 같은 reward를 사용합니다."""
)


code(
    """def make_env(features, returns):
    return PortfolioAllocationEnv(features, returns)


def train_rl_model(algo_name, X_train, R_train, seed=SEED):
    env = DummyVecEnv([lambda: make_env(X_train, R_train)])
    policy_kwargs = {"net_arch": NET_ARCH}

    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            seed=seed,
            verbose=0,
        )
    elif algo_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=7e-4,
            n_steps=16,
            gamma=0.99,
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError(f"unknown algo_name: {algo_name}")

    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    return model


def predict_weight_path(model, X, R):
    env = make_env(X, R)
    obs, _ = env.reset()
    rows = []

    for _ in range(len(R)):
        action, _ = model.predict(obs, deterministic=True)
        rows.append(softmax_action(action))
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    return pd.DataFrame(rows, index=R.index[: len(rows)], columns=ASSETS)


print("학습/예측 함수 준비 완료")"""
)


md(
    """## 앙상블 전략 함수

앙상블은 PPO와 A2C가 각각 낸 자산 비중을 다시 섞습니다.

```python
final_weight = ppo_ratio * ppo_weight + (1 - ppo_ratio) * a2c_weight
```

자동 전략은 현재 시점의 수익률을 보기 전에, 이전 시점까지 누적된 PPO/A2C 실현 수익으로 다음 비율을 정합니다."""
)


code(
    """STRATEGIES = [
    {"name": "PPO_Only", "kind": "fixed", "fixed_ppo": 1.00},
    {"name": "A2C_Only", "kind": "fixed", "fixed_ppo": 0.00},
    {"name": "Manual_PPO25_A2C75", "kind": "fixed", "fixed_ppo": 0.25},
    {"name": "Manual_PPO50_A2C50", "kind": "fixed", "fixed_ppo": 0.50},
    {"name": "Manual_PPO75_A2C25", "kind": "fixed", "fixed_ppo": 0.75},
    {"name": "Auto_RollingSharpe", "kind": "auto_sharpe"},
    {"name": "Auto_DD_Guard", "kind": "auto_dd_guard"},
    {"name": "Manual_VIX_Override", "kind": "vix_override", "fixed_ppo": 0.50},
    {"name": "Auto_HMM_Regime", "kind": "hmm_regime"},
]


def rolling_sharpe_score(values, annualization=ANNUALIZATION):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    return float(arr.mean() / (arr.std() + 1e-8) * np.sqrt(annualization))


def ratio_from_scores(ppo_score, a2c_score, temperature=1.0):
    scores = np.array([ppo_score, a2c_score], dtype=np.float64) / temperature
    scores = np.clip(scores, -10, 10)
    exp_scores = np.exp(scores - scores.max())
    return float(exp_scores[0] / exp_scores.sum())


def run_ensemble_strategy(
    ppo_weights,
    a2c_weights,
    returns,
    raw_signals,
    strategy,
    vix_threshold=None,
    transaction_cost=TRANSACTION_COST,
    X_test=None,
):
    ppo_hist = deque(maxlen=SHARPE_WINDOW)
    a2c_hist = deque(maxlen=SHARPE_WINDOW)

    prev_w = np.ones(len(ASSETS)) / len(ASSETS)
    port_val = 1.0
    peak_val = 1.0

    ratio_rows = []
    weight_rows = []
    net_returns = []
    gross_returns = []
    turnovers = []
    costs = []
    dd_rows = []

    for dt in returns.index:
        ppo_w = ppo_weights.loc[dt].values
        a2c_w = a2c_weights.loc[dt].values

        kind = strategy["kind"]
        if kind == "fixed":
            ppo_ratio = strategy["fixed_ppo"]
        elif kind in ["auto_sharpe", "auto_dd_guard"]:
            ppo_score = rolling_sharpe_score(ppo_hist)
            a2c_score = rolling_sharpe_score(a2c_hist)
            ppo_ratio = ratio_from_scores(ppo_score, a2c_score)

            if kind == "auto_dd_guard":
                current_dd = (peak_val - port_val) / (peak_val + 1e-8)
                if current_dd > 0.10:
                    ppo_ratio = min(ppo_ratio, 0.25)
        elif kind == "vix_override":
            ppo_ratio = strategy.get("fixed_ppo", 0.5)
            if vix_threshold is not None and raw_signals.loc[dt, "VIX"] > vix_threshold:
                ppo_ratio = 0.0
        elif kind == "hmm_regime":
            if X_test is not None:
                low_vol_prob = X_test.loc[dt, ["HMM_Prob_0", "HMM_Prob_1", "HMM_Prob_2"]].sum()
                ppo_ratio = float(low_vol_prob)
            else:
                ppo_ratio = 0.5
        else:
            raise ValueError(f"unknown strategy kind: {kind}")

        final_w = ppo_ratio * ppo_w + (1.0 - ppo_ratio) * a2c_w
        final_w = final_w / (final_w.sum() + 1e-12)

        r = returns.loc[dt].values
        ppo_model_ret = float(np.dot(ppo_w, r))
        a2c_model_ret = float(np.dot(a2c_w, r))
        gross_ret = float(np.dot(final_w, r))
        turnover = float(np.sum(np.abs(final_w - prev_w)) / 2.0)
        cost = turnover * transaction_cost
        net_ret = gross_ret - cost

        port_val *= 1.0 + net_ret
        peak_val = max(peak_val, port_val)
        current_dd = (peak_val - port_val) / (peak_val + 1e-8)

        ppo_hist.append(ppo_model_ret)
        a2c_hist.append(a2c_model_ret)
        prev_w = final_w

        ratio_rows.append(ppo_ratio)
        weight_rows.append(final_w)
        gross_returns.append(gross_ret)
        net_returns.append(net_ret)
        turnovers.append(turnover)
        costs.append(cost)
        dd_rows.append(current_dd)

    weight_df = pd.DataFrame(weight_rows, index=returns.index, columns=ASSETS)
    result = {
        "strategy": strategy["name"],
        "ppo_ratio": pd.Series(ratio_rows, index=returns.index, name="ppo_ratio"),
        "weights": weight_df,
        "gross_returns": pd.Series(gross_returns, index=returns.index, name="gross_return"),
        "net_returns": pd.Series(net_returns, index=returns.index, name="net_return"),
        "turnover": pd.Series(turnovers, index=returns.index, name="turnover"),
        "cost": pd.Series(costs, index=returns.index, name="cost"),
        "drawdown": pd.Series(dd_rows, index=returns.index, name="drawdown"),
    }
    result["equity"] = (1.0 + result["net_returns"]).cumprod()
    return result


print("앙상블 전략 준비 완료")"""
)


md(
    """## 성과 지표"""
)


code(
    """def max_drawdown(equity):
    eq = pd.Series(equity).dropna()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def perf_metrics(returns, turnover=None, annualization=ANNUALIZATION):
    ret = pd.Series(returns).dropna()
    equity = (1.0 + ret).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    years = len(ret) / annualization
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(ret.std() * np.sqrt(annualization))
    sharpe = float(ret.mean() / (ret.std() + 1e-8) * np.sqrt(annualization))
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


print("성과 지표 함수 준비 완료")"""
)


md(
    """## 전체 실험 실행

시간이 오래 걸리는 셀입니다. 기본 설정 기준:

- fold마다 PPO 1개 + A2C 1개 학습
- 각 모델 `300,000` timesteps
- 결과는 `ensemble_results/`에 CSV로 저장
- 모델은 `ensemble_models/`에 zip으로 저장"""
)


code(
    """fold_rows = []
oos_return_parts = {s["name"]: [] for s in STRATEGIES}
oos_turnover_parts = {s["name"]: [] for s in STRATEGIES}
oos_ratio_parts = {s["name"]: [] for s in STRATEGIES}
oos_weight_parts = {s["name"]: [] for s in STRATEGIES}

for split in splits:
    fold = split["fold"]
    print("=" * 90)
    print(
        f"Fold {fold}: "
        f"Train {split['train_start'].date()} ~ {split['train_end'].date()} | "
        f"Test {split['test_start'].date()} ~ {split['test_end'].date()}"
    )

    train_mask = split["train_mask"]
    test_mask = split["test_mask"]

    scaler = StandardScaler()
    X_train_base = pd.DataFrame(
        scaler.fit_transform(X_all.loc[train_mask]),
        index=X_all.loc[train_mask].index,
        columns=FEATURES,
    )
    X_test_base = pd.DataFrame(
        scaler.transform(X_all.loc[test_mask]),
        index=X_all.loc[test_mask].index,
        columns=FEATURES,
    )

    R_train_base = R_all.loc[train_mask]
    R_test_base = R_all.loc[test_mask]

    X_train, X_test, R_train, R_test = append_hmm_probs(
        X_train_base, X_test_base, R_train_base, R_test_base, split
    )

    signals_train = raw_signal_all.loc[X_train.index]
    signals_test = raw_signal_all.loc[X_test.index]
    vix_threshold = float(signals_train["VIX"].quantile(0.80))

    print(f"  VIX override threshold(train 80%): {vix_threshold:.2f}")
    print("  PPO 학습 중...")
    ppo_model = train_rl_model("PPO", X_train, R_train, seed=SEED + fold)
    print("  A2C 학습 중...")
    a2c_model = train_rl_model("A2C", X_train, R_train, seed=SEED + 100 + fold)

    if SAVE_MODELS:
        ppo_model.save(os.path.join(MODEL_DIR, f"fold{fold}_ppo"))
        a2c_model.save(os.path.join(MODEL_DIR, f"fold{fold}_a2c"))

    print("  테스트 기간 weight 예측 중...")
    ppo_weights = predict_weight_path(ppo_model, X_test, R_test)
    a2c_weights = predict_weight_path(a2c_model, X_test, R_test)

    for strategy in STRATEGIES:
        res = run_ensemble_strategy(
            ppo_weights=ppo_weights,
            a2c_weights=a2c_weights,
            returns=R_test,
            raw_signals=signals_test,
            strategy=strategy,
            vix_threshold=vix_threshold,
            X_test=X_test,
        )
        m = perf_metrics(res["net_returns"], res["turnover"])
        fold_rows.append(
            {
                "fold": fold,
                "strategy": strategy["name"],
                "test_start": split["test_start"].date(),
                "test_end": split["test_end"].date(),
                **m,
                "mean_ppo_ratio": float(res["ppo_ratio"].mean()),
                "vix_threshold": vix_threshold,
            }
        )
        oos_return_parts[strategy["name"]].append(res["net_returns"])
        oos_turnover_parts[strategy["name"]].append(res["turnover"])
        oos_ratio_parts[strategy["name"]].append(res["ppo_ratio"])
        oos_weight_parts[strategy["name"]].append(res["weights"])

    fold_df = pd.DataFrame([r for r in fold_rows if r["fold"] == fold])
    display(
        fold_df.sort_values("sharpe", ascending=False)[
            ["strategy", "total_return", "sharpe", "mdd", "calmar", "avg_turnover", "mean_ppo_ratio"]
        ]
    )

fold_results = pd.DataFrame(fold_rows)
fold_results.to_csv(os.path.join(RESULT_DIR, "fold_results.csv"), index=False, encoding="utf-8-sig")

print("실험 완료")
display(fold_results.head())"""
)


md(
    """## OOS 통합 결과"""
)


code(
    """aggregate_rows = []
oos_returns = {}
oos_turnovers = {}
oos_ratios = {}
oos_weights = {}

for strategy in [s["name"] for s in STRATEGIES]:
    ret = pd.concat(oos_return_parts[strategy]).sort_index()
    turnover = pd.concat(oos_turnover_parts[strategy]).sort_index()
    ratio = pd.concat(oos_ratio_parts[strategy]).sort_index()
    weights = pd.concat(oos_weight_parts[strategy]).sort_index()

    oos_returns[strategy] = ret
    oos_turnovers[strategy] = turnover
    oos_ratios[strategy] = ratio
    oos_weights[strategy] = weights

    m = perf_metrics(ret, turnover)
    aggregate_rows.append({"strategy": strategy, **m, "mean_ppo_ratio": float(ratio.mean())})

summary = pd.DataFrame(aggregate_rows).sort_values(["sharpe", "calmar"], ascending=False)
summary.to_csv(os.path.join(RESULT_DIR, "summary.csv"), index=False, encoding="utf-8-sig")

display(
    summary.assign(
        total_return=lambda x: x["total_return"] * 100,
        cagr=lambda x: x["cagr"] * 100,
        vol=lambda x: x["vol"] * 100,
        mdd=lambda x: x["mdd"] * 100,
    ).round(3)
)

best_strategy = summary.iloc[0]["strategy"]
print(f"최고 Sharpe 전략: {best_strategy}")"""
)


md(
    """## 시각화 및 결과 저장"""
)


code(
    """equity_df = pd.DataFrame({name: (1.0 + ret).cumprod() for name, ret in oos_returns.items()})
ratio_df = pd.DataFrame(oos_ratios)
turnover_df = pd.DataFrame(oos_turnovers)

fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})

equity_df.plot(ax=axes[0], linewidth=2)
axes[0].set_title("A2C + PPO 앙상블 전략별 OOS 누적 수익")
axes[0].set_ylabel("Portfolio Value")
axes[0].grid(alpha=0.3)

plot_summary = summary.set_index("strategy")
plot_summary["sharpe"].sort_values().plot(kind="barh", ax=axes[1], color="#357ABD")
axes[1].set_title("전략별 Sharpe")
axes[1].grid(axis="x", alpha=0.3)

(plot_summary["mdd"].sort_values() * 100).plot(kind="barh", ax=axes[2], color="#C44E52")
axes[2].set_title("전략별 MDD (%)")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "ensemble_strategy_comparison.png"), dpi=160, bbox_inches="tight")
plt.show()

plt.figure(figsize=(14, 5))
for col in ["Auto_RollingSharpe", "Auto_DD_Guard", "Manual_VIX_Override", "Auto_HMM_Regime"]:
    if col in ratio_df.columns:
        ratio_df[col].plot(label=col, linewidth=2)
plt.title("동적 전략의 PPO 비율 변화")
plt.ylabel("PPO ratio")
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "dynamic_ppo_ratio.png"), dpi=160, bbox_inches="tight")
plt.show()

equity_df.to_csv(os.path.join(RESULT_DIR, "oos_equity_curves.csv"), encoding="utf-8-sig")
ratio_df.to_csv(os.path.join(RESULT_DIR, "oos_ppo_ratios.csv"), encoding="utf-8-sig")
turnover_df.to_csv(os.path.join(RESULT_DIR, "oos_turnover.csv"), encoding="utf-8-sig")

best_weights = oos_weights[best_strategy]
best_weights.to_csv(os.path.join(RESULT_DIR, f"best_weights_{best_strategy}.csv"), encoding="utf-8-sig")

print("저장 완료")
print(f"- {RESULT_DIR}/summary.csv")
print(f"- {RESULT_DIR}/fold_results.csv")
print(f"- {RESULT_DIR}/ensemble_strategy_comparison.png")
print(f"- {RESULT_DIR}/dynamic_ppo_ratio.png")
print(f"- {RESULT_DIR}/oos_equity_curves.csv")
print(f"- {RESULT_DIR}/oos_ppo_ratios.csv")
print(f"- {RESULT_DIR}/oos_turnover.csv")
print(f"- {RESULT_DIR}/best_weights_{best_strategy}.csv")"""
)


md(
    """## 확인 포인트

1. `summary.csv`에서 Sharpe, MDD, Calmar를 비교합니다.
2. `fold_results.csv`에서 특정 연도에만 좋은 전략인지 확인합니다.
3. `dynamic_ppo_ratio.png`에서 자동 비율이 지나치게 자주 흔들리는지 확인합니다.
4. 최종 제출에는 `Manual_PPO50_A2C50` 같은 단순 기준선 대비 `Auto_RollingSharpe`, `Auto_DD_Guard`, `Manual_VIX_Override`가 개선됐는지 정리하면 됩니다."""
)


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

for path in [OUTPUT_NB]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"노트북 생성 완료: {OUTPUT_NB}")
print(f"총 셀 수: {len(cells)}")
