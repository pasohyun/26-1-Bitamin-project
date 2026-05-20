# BITAmin 금융시계열 2조

> 마지막 정리: 2026-05-20 (KST)  
> GitHub 추적 기준: `sohyun` 브랜치, 커밋 `dee9755`

## 프로젝트 개요

주식 입문자를 위한 **거시 국면 기반 자산배분 학습 게임** 프로젝트입니다.  
핵심 아이디어는 다음과 같습니다.

1. 시장 데이터를 수집하고
2. HMM으로 6개 국면을 추정한 뒤
3. PPO/A2C 기반 자산배분 모델을 만들고
4. 사용자가 AI와 1개월 수익률 승부를 하는 Streamlit 게임으로 연결합니다.

---

## GitHub에 올라간 파일(추적 파일) 전체

아래는 `git ls-files` 기준으로 현재 GitHub 추적 대상 파일을 빠짐없이 정리한 목록입니다.

### 1) 앱/스크립트/노트북

- `README.md`
- `requirements.txt`
- `environment.yml`
- `app.py`
- `select_data.ipynb`
- `HMM_Final.ipynb`
- `A2C_PPO_Ensemble_Experiment.ipynb`
- `build_ensemble_compare_nb.py`
- `final_hmm_rf_proxy_pipeline.py`
- `ppo_team_reward_fixed.py`
- `refine_ensemble_strategies.py`
- `update_ipynb.py`

### 2) 데이터/모델 아티팩트

- `cached_raw_data.csv`
- `final_hmm_rf_outputs/hmm_labeled_data_target_style_seed42.csv`
- `ppo_v6_v6_monthly_final.zip`
- `ensemble_models/fold1_a2c.zip`, `ensemble_models/fold1_ppo.zip`
- `ensemble_models/fold2_a2c.zip`, `ensemble_models/fold2_ppo.zip`
- `ensemble_models/fold3_a2c.zip`, `ensemble_models/fold3_ppo.zip`
- `ensemble_models/fold4_a2c.zip`, `ensemble_models/fold4_ppo.zip`
- `ensemble_models/fold5_a2c.zip`, `ensemble_models/fold5_ppo.zip`
- `ensemble_models/fold6_a2c.zip`, `ensemble_models/fold6_ppo.zip`
- `ensemble_models/fold7_a2c.zip`, `ensemble_models/fold7_ppo.zip`
- `ensemble_models/fold8_a2c.zip`, `ensemble_models/fold8_ppo.zip`
- `ensemble_models/fold9_a2c.zip`, `ensemble_models/fold9_ppo.zip`
- `ensemble_models/fold10_a2c.zip`, `ensemble_models/fold10_ppo.zip`
- `ensemble_models/fold11_a2c.zip`, `ensemble_models/fold11_ppo.zip`
- `ensemble_models/fold12_a2c.zip`, `ensemble_models/fold12_ppo.zip`
- `ensemble_models/fold13_a2c.zip`, `ensemble_models/fold13_ppo.zip`
- `ensemble_models/fold14_a2c.zip`, `ensemble_models/fold14_ppo.zip`

---

## 현재 구현 범위

| 단계 | 상태 | 주요 파일 |
|---|---|---|
| 데이터 수집/정제 | 구현 | `select_data.ipynb`, `cached_raw_data.csv` |
| HMM 6국면 + RF Proxy | 구현 | `HMM_Final.ipynb`, `final_hmm_rf_proxy_pipeline.py` |
| PPO 자산배분 베이스라인 | 구현 | `ppo_team_reward_fixed.py`, `ppo_v6_v6_monthly_final.zip` |
| A2C+PPO 앙상블 | 구현 | `A2C_PPO_Ensemble_Experiment.ipynb`, `refine_ensemble_strategies.py`, `ensemble_models/` |
| 게임 UI (Streamlit) | 구현 | `app.py` |

---

## 데이터 스냅샷

`cached_raw_data.csv` 기준:

- 기간: **2009-01-02 ~ 2026-04-01**
- 행 수: **4,338행**
- 열 수: **24개 데이터 컬럼** (인덱스 날짜 제외)
- 자산 예시: `SPY`, `TLT`, `SHV`, `GLD`, `DBC`, `QQQ`, `EEM`, `VNQ`
- 거시 예시: `VIX`, `DXY`, `US10Y`, `US2Y`, `US3M`, `Jobless_Claims`

---

## 핵심 모듈 설명

### 1) `app.py` (게임 앱)

- Streamlit 기반 6회차 라운드 게임
- 플레이어 vs AI 패널 P(룰 기반 + 국면 기반 가중치 조절)
- 사용 자산: `SPY`, `TLT`, `SHV`, `GLD`, `DBC`
- 입력: 1개월 시그널 창(거시/시장 지표)
- 출력: 플레이어/AI 수익률 비교, 점수, 누적 상금판(회차당 ±500,000원)
- 데이터 로딩 우선순위
  - `cached_raw_data.csv` 로드
  - `final_hmm_rf_outputs/hmm_labeled_data_target_style_seed42.csv`가 있으면 HMM Level 사용
  - 없으면 VIX 분위수 기반 fallback Level 사용

### 2) `final_hmm_rf_proxy_pipeline.py`

- HMM 6국면 학습 + RF Proxy 중요도 분석 파이프라인
- HMM 핵심 8센서:
  - `Equity_vs_Bond`, `US10Y`, `Jobless_Claims_MA`, `VIX`
  - `Market_Breadth`, `spread_10y2y`, `DXY`, `Copper_Gold_Ratio`
- RF Proxy는 23개 피처 중요도 산출
- 출력 예시:
  - `hmm_6_regimes_*.png`
  - `rf_proxy_23_sensors_*.png`
  - `hmm_level_means_*.csv`
  - `rf_proxy_importance_*.csv`
  - `hmm_labeled_data_*.csv`

### 3) `ppo_team_reward_fixed.py`

- PPO 베이스라인 + walk-forward(4년 train / 1년 test)
- 주요 보정 사항 반영:
  - look-ahead bias 제거 (`t` 관측으로 `t+1` 수익 사용)
  - scaler/HMM을 fold train 구간에서만 fit
  - 거래비용/turnover 패널티 반영
- 결과 저장:
  - `ppo_baseline_portfolio_value.csv`
  - `ppo_baseline_trade_info.csv`

### 4) `A2C_PPO_Ensemble_Experiment.ipynb` / `refine_ensemble_strategies.py`

- A2C/PPO 혼합 비율 실험(고정/동적 전략)
- `ensemble_models/`의 fold별 저장 모델(`fold{n}_ppo.zip`, `fold{n}_a2c.zip`)을 로드해 OOS 성과 비교
- `refine_ensemble_strategies.py` 결과 저장:
  - `ensemble_refined_results/refined_summary.csv`
  - `ensemble_refined_results/refined_fold_results.csv`
  - `ensemble_refined_results/refined_strategy_comparison.png`
  - `ensemble_refined_results/refined_dynamic_ppo_ratio.png`

### 5) `build_ensemble_compare_nb.py`

- 앙상블 실험 노트북(`A2C_PPO_Ensemble_Experiment.ipynb`) 생성/재빌드 스크립트

---

## 앙상블 결과 요약 (로컬 산출물 기준)

`ensemble_refined_results/refined_summary.csv` 상위 Sharpe 전략:

| 전략 | Total Return | CAGR | Vol | Sharpe | MDD | Mean PPO Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `Vol_Target_DD_Guard` | 115.36% | 6.08% | 10.36% | 0.623 | -19.20% | 36.04% |
| `Turnover_Aware_DD_Guard` | 115.69% | 6.09% | 10.42% | 0.621 | -20.34% | 41.10% |
| `Auto_DD_Guard_v1` | 112.96% | 5.99% | 10.32% | 0.616 | -19.41% | 37.36% |

---

## 실행 방법

### 1) 패키지 설치

`pip` 또는 `conda` 중 하나를 선택해 설치합니다.

```bash
pip install -r requirements.txt
```

또는

```bash
conda env create -f environment.yml
conda activate bitamin-quant
```


### 2) HMM + RF Proxy 산출

```bash
python final_hmm_rf_proxy_pipeline.py --cache cached_raw_data.csv --outdir final_hmm_rf_outputs --equity-bond-mode target_style --seed 42
```

### 3) PPO 베이스라인 실행

```bash
python ppo_team_reward_fixed.py
```

> 실행 전 `FRED_API_KEY` 환경변수를 설정해야 합니다.

### 4) 앙상블 재평가

```bash
python refine_ensemble_strategies.py
```

### 5) 게임 실행

```bash
streamlit run app.py
```

---

## 참고/주의

1. 노트북/스크립트는 FRED API 키를 하드코딩하지 않고 `FRED_API_KEY` 환경변수를 사용합니다.
2. `build_ensemble_compare_nb.py`는 노트북 본문을 재생성하므로, 수동 수정본이 있으면 백업 후 실행 권장.
3. `app.py`는 외부 이미지 URL을 사용하지 않으며, 로컬 이미지가 없으면 이미지 없이 렌더링합니다.

---

## 한 줄 요약

이 저장소는 **HMM 6국면 해석 + PPO/A2C 자산배분 + Streamlit 대결형 학습 UI**를 한 프로젝트로 묶은, 주식 입문자용 금융 시계열 학습 게임 구현체입니다.
