from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "cached_raw_data.csv"
HMM_DATA_PATH = BASE_DIR / "final_hmm_rf_outputs" / "hmm_labeled_data_target_style_seed42.csv"

LOOKBACK_DAYS = 21
HORIZON_DAYS = 21
SIGNAL_GEM_VALUE = 500_000

# ── 이미지 설정 ────────────────────────────────────────────────────────────────
# 로컬 파일이 있으면 우선 사용, 없으면 Unsplash 고화질 URL 사용
STUDIO_IMAGE_CANDIDATES = [
    BASE_DIR / "assets" / "signal_house.jpg",
    BASE_DIR / "assets" / "signal_house.png",
    BASE_DIR / "assets" / "signal_house.webp",
]
# 하트시그널 감성 — 세련된 거실/패널룸 분위기
STUDIO_IMAGE_FALLBACK = "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=1200&q=85"

# ETF 출연진 프로필 사진 — 이모지 플레이스홀더 사용 (저작권 문제로 사진 미사용)
ASSET_PHOTOS: dict[str, str] = {}

# 국면별 분위기 이미지
REGIME_PHOTOS = {
    1: "https://images.unsplash.com/photo-1504703395950-b89145a5425b?auto=format&fit=crop&w=800&q=80",
    2: "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?auto=format&fit=crop&w=800&q=80",
    3: "https://images.unsplash.com/photo-1529333166437-7750a6dd5a70?auto=format&fit=crop&w=800&q=80",
    4: "https://images.unsplash.com/photo-1499343162891-ad3ef3a53e57?auto=format&fit=crop&w=800&q=80",
    5: "https://images.unsplash.com/photo-1583511655826-05700d52f4d9?auto=format&fit=crop&w=800&q=80",
    6: "https://images.unsplash.com/photo-1518199266791-5375a83190b7?auto=format&fit=crop&w=800&q=80",
}

ASSETS = ["SPY", "TLT", "SHV", "GLD", "DBC"]

ASSET_META = {
    "SPY": {
        "name": "직진남",
        "tagline": "분위기 좋으면 바로 치고 들어오는 메인 출연자",
        "role": "미국 대표 주식",
        "profile": "상승장에서는 가장 먼저 마음을 표현하는 타입입니다. 모두의 관심을 받지만 시장 분위기가 나빠지면 감정 기복도 커집니다.",
        "color": "#d84f73",
        "emoji": "💘",
    },
    "TLT": {
        "name": "돈많은 연하남",
        "tagline": "위기 때 갑자기 든든해지는 장기 국채남",
        "role": "장기 국채",
        "profile": "평소에는 조용하지만 불안한 회차에서 존재감이 커지는 타입입니다. 다만 금리 변화에 민감해서 타이밍을 잘 봐야 합니다.",
        "color": "#4f7fb8",
        "emoji": "🫶",
    },
    "SHV": {
        "name": "무해한 집돌이",
        "tagline": "설렘은 적지만 불안할 때 제일 편한 사람",
        "role": "단기 국채",
        "profile": "큰 이벤트는 없지만 마음이 편한 현금성 출연자입니다. 패닉 회차에서는 손실을 줄여주는 안전한 대기실 역할을 합니다.",
        "color": "#687385",
        "emoji": "🏠",
    },
    "GLD": {
        "name": "신비주의 연상남",
        "tagline": "평소엔 조용한데 위기 때 자꾸 생각나는 사람",
        "role": "금",
        "profile": "시장 불확실성이 커질수록 묘하게 끌리는 안전자산 캐릭터입니다. 모두가 흔들릴 때 혼자 다른 분위기를 냅니다.",
        "color": "#c69b3f",
        "emoji": "✨",
    },
    "DBC": {
        "name": "핫한 자유영혼",
        "tagline": "물가와 원자재 뉴스에 바로 반응하는 예측불가 출연자",
        "role": "원자재",
        "profile": "인플레이션 회차에 갑자기 매력이 올라오는 타입입니다. 자유분방해서 예측은 어렵지만 존재감은 확실합니다.",
        "color": "#58a98b",
        "emoji": "🌊",
    },
}

REGIMES = {
    1: {
        "finance": "Accumulation",
        "episode": "EP.01 첫 입주, 조용한 눈빛",
        "signal": "하락세가 멈추고 VIX가 안정됩니다. 시장이 횡보하면서 반등을 준비하는 구간입니다.",
        "panel": "아직 확실한 러브라인은 없지만 누군가에게 시선이 머무르기 시작합니다.",
        "guide": "공격형 ETF를 조금씩 살피되 방어형 ETF도 유지하는 전략이 어울립니다.",
        "intro": "첫 입주가 시작됐습니다. 아직 모두가 조심스럽지만 시장의 큰 하락은 멈췄고 조심스러운 매수 시그널이 나타납니다.",
        "risk": 28,
        "mood": "🌱 조심스러운 탐색",
        "badge_color": "#58a98b",
    },
    2: {
        "finance": "Expansion",
        "episode": "EP.02 확신의 시그널",
        "signal": "VIX가 낮고 주가가 상승하며 위험자산 선호가 뚜렷합니다.",
        "panel": "대화도 잘 통하고 데이트 분위기도 좋습니다. 시장의 마음이 직진남 수피에게 향합니다.",
        "guide": "SPY 같은 성장형 자산 비중을 확대하는 전략이 유리할 수 있습니다.",
        "intro": "오늘은 눈빛이 다릅니다. 시장은 위험자산에게 분명한 호감을 보내고 있습니다.",
        "risk": 18,
        "mood": "💕 설레는 상승",
        "badge_color": "#d84f73",
    },
    3: {
        "finance": "Euphoria",
        "episode": "EP.03 과몰입 러브라인",
        "signal": "상승세가 강하지만 과열 신호가 등장합니다. VIX가 바닥권에서 미세하게 반등할 수 있습니다.",
        "panel": "모두가 확정 러브라인이라고 믿는 순간입니다. 하지만 반전 가능성도 함께 커집니다.",
        "guide": "공격형 자산은 일부 유지하되 SHV, GLD, 채권으로 방어를 시작하는 구간입니다.",
        "intro": "패널석이 술렁입니다. 분위기는 좋지만 시장도 너무 뜨거워졌습니다.",
        "risk": 44,
        "mood": "🔥 과열 주의보",
        "badge_color": "#e67e22",
    },
    4: {
        "finance": "Warning",
        "episode": "EP.04 엇갈리는 문자",
        "signal": "금리가 상승하고 VIX가 반등하며 주가 모멘텀이 둔화됩니다.",
        "panel": "문자 선택이 엇갈리고 패널들이 이상한 기류를 감지합니다.",
        "guide": "SPY 비중을 줄이고 SHV, GLD, TLT 같은 방어 자산을 늘리는 전략이 필요합니다.",
        "intro": "어제와는 다른 문자 선택이 나왔습니다. 겉보기엔 괜찮지만 미묘한 불안이 커집니다.",
        "risk": 62,
        "mood": "😰 불안한 기류",
        "badge_color": "#c79a3d",
    },
    5: {
        "finance": "Crisis",
        "episode": "EP.05 최종 선택의 대혼란",
        "signal": "VIX가 폭등하고 위험자산이 급락합니다. 시장의 마음이 공격형 자산에서 급격히 이탈합니다.",
        "panel": "예상했던 러브라인이 무너지고 스튜디오가 뒤집힙니다.",
        "guide": "수익보다 생존이 우선입니다. SHV, GLD, TLT 중심의 방어가 중요합니다.",
        "intro": "스튜디오가 뒤집혔습니다. 모두가 예상한 러브라인이 무너지고 시장에서도 패닉이 발생했습니다.",
        "risk": 92,
        "mood": "🚨 패닉 발생",
        "badge_color": "#b93462",
    },
    6: {
        "finance": "Bottoming",
        "episode": "EP.06 새로운 시그널",
        "signal": "VIX가 고점에서 내려오고 공포가 완화됩니다. 반등 준비가 나타나는 구간입니다.",
        "panel": "혼란이 지나가고 새로운 시선이 포착됩니다.",
        "guide": "방어 자산에서 SPY로 점진 이동하며 반등 기회를 살피는 전략이 어울립니다.",
        "intro": "긴 혼란이 지나가고 새로운 시그널이 포착됩니다. 아직 확신하긴 이르지만 시선이 다시 움직입니다.",
        "risk": 54,
        "mood": "🌅 새벽의 시그널",
        "badge_color": "#4f7fb8",
    },
}

RECOMMENDED_WEIGHTS = {
    1: {"SPY": 0.40, "TLT": 0.20, "SHV": 0.20, "GLD": 0.10, "DBC": 0.10},
    2: {"SPY": 0.60, "TLT": 0.10, "SHV": 0.10, "GLD": 0.10, "DBC": 0.10},
    3: {"SPY": 0.45, "TLT": 0.15, "SHV": 0.15, "GLD": 0.15, "DBC": 0.10},
    4: {"SPY": 0.25, "TLT": 0.25, "SHV": 0.25, "GLD": 0.15, "DBC": 0.10},
    5: {"SPY": 0.10, "TLT": 0.30, "SHV": 0.30, "GLD": 0.25, "DBC": 0.05},
    6: {"SPY": 0.45, "TLT": 0.20, "SHV": 0.15, "GLD": 0.10, "DBC": 0.10},
}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="마켓시그널 💘📈", layout="wide", page_icon="💘")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Myeongjo:wght@400;700;800&family=Pretendard:wght@300;400;500;600;700;800;900&family=Noto+Sans+KR:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg-start: #fff5f7;
        --bg-mid: #fef0f4;
        --bg-end: #fde8ef;
        --panel: rgba(255,255,255,.72);
        --panel-hover: rgba(255,255,255,.90);
        --panel-strong: rgba(255,255,255,.85);
        --ink: #3a1a28;
        --ink-muted: rgba(58,26,40,.65);
        --ink-dim: rgba(58,26,40,.42);
        --pink: #e8426e;
        --pink-light: #f483a2;
        --pink-deep: #c2245a;
        --teal: #3db89e;
        --gold: #c9873a;
        --blue: #5a8fc4;
        --purple: #9060c0;
        --line: rgba(232,66,110,.22);
        --line-soft: rgba(232,66,110,.10);
        --shadow: 0 16px 48px rgba(200,80,120,.12);
        --glow-pink: 0 0 32px rgba(232,66,110,.12);
        --glow-teal: 0 0 32px rgba(61,184,158,.10);
        --radius: 14px;
        --radius-sm: 9px;
    }

    /* ── 전체 배경 ── */
    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at 10% 0%, rgba(255,200,215,.55), transparent),
            radial-gradient(ellipse 60% 50% at 90% 10%, rgba(255,230,240,.60), transparent),
            radial-gradient(ellipse 70% 50% at 50% 100%, rgba(255,210,225,.35), transparent),
            linear-gradient(160deg, #fff5f7 0%, #fef0f4 50%, #fde8ef 100%);
        color: var(--ink);
        font-family: "Pretendard", "Noto Sans KR", system-ui, sans-serif;
        min-height: 100vh;
    }

    /* ── 헤더/사이드바 ── */
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,240,245,.97), rgba(254,232,240,.97)) !important;
        border-right: 1px solid var(--line) !important;
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }
    .stApp button, .stApp input, .stApp textarea, .stApp select {
        font-family: "Pretendard", "Noto Sans KR", system-ui, sans-serif !important;
    }
    .block-container {
        max-width: 1380px !important;
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
    }

    /* ── 타이포 ── */
    h1, h2, h3, .serif {
        font-family: "Nanum Myeongjo", serif !important;
        letter-spacing: -0.02em !important;
    }
    h1, h2, h3 { color: var(--ink) !important; }

    /* ── 시스템바 ── */
    .sys-bar {
        display: flex;
        gap: 8px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    .sys-chip {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 5px 12px;
        font-size: 11px;
        font-weight: 700;
        color: var(--pink-light);
        letter-spacing: .06em;
        text-transform: uppercase;
        backdrop-filter: blur(8px);
    }

    /* ── 히어로 배너 ── */
    .hero-banner {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 0;
        box-shadow: var(--shadow), var(--glow-pink);
        margin-bottom: 20px;
        min-height: 220px;
    }
    .hero-bg {
        position: absolute; inset: 0;
        background-size: cover;
        background-position: center;
        filter: brightness(.38) saturate(.8);
    }
    .hero-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(135deg,
            rgba(255,240,245,.88) 0%,
            rgba(255,230,238,.60) 55%,
            rgba(255,220,235,.40) 100%);
    }
    .hero-grain {
        position: absolute; inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
        opacity: .45;
        pointer-events: none;
    }
    .hero-inner {
        position: relative; z-index: 2;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 24px;
        align-items: center;
        padding: 28px 32px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--pink-deep);
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
        color: #fff;
        margin-bottom: 10px;
        box-shadow: 0 4px 16px rgba(194,48,96,.4);
    }
    .hero-title {
        font-family: "Nanum Myeongjo", serif;
        font-size: 42px;
        font-weight: 800;
        line-height: 1.12;
        color: var(--ink);
        margin-bottom: 10px;
        text-shadow: 0 1px 8px rgba(200,80,120,.15);
    }
    .hero-sub {
        font-size: 14px;
        line-height: 1.65;
        color: var(--ink-muted);
        max-width: 640px;
    }
    .hero-risk-wrap { margin-top: 16px; }
    .hero-risk-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--ink-dim);
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-risk-track {
        height: 6px;
        border-radius: 999px;
        background: rgba(255,255,255,.12);
        overflow: hidden;
        width: 320px;
        max-width: 100%;
    }
    .hero-risk-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--teal), var(--gold), var(--pink));
        transition: width .6s cubic-bezier(.22,1,.36,1);
    }
    .hero-photo-wrap {
        width: 210px;
        height: 180px;
        border-radius: var(--radius);
        overflow: hidden;
        border: 1px solid rgba(240,96,138,.28);
        box-shadow: 0 20px 48px rgba(0,0,0,.6);
        flex-shrink: 0;
        position: relative;
    }
    .hero-photo-wrap img {
        width: 100%; height: 100%;
        object-fit: cover;
        filter: saturate(.85) brightness(.9);
    }
    .hero-photo-tag {
        position: absolute;
        bottom: 8px; left: 8px;
        background: rgba(0,0,0,.6);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .06em;
        color: #fff;
        backdrop-filter: blur(6px);
    }

    /* ── 시그널 카드 그리드 ── */
    .signal-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 4px 0 18px;
    }
    .signal-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 14px;
        backdrop-filter: blur(10px);
        transition: border-color .2s, background .2s;
        min-height: 130px;
    }
    .signal-card:hover {
        background: var(--panel-hover);
        border-color: var(--line);
    }
    .signal-card.warn { border-color: rgba(240,96,138,.35); }
    .signal-card.safe { border-color: rgba(95,212,188,.25); }
    .signal-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .05em;
        text-transform: uppercase;
        color: var(--ink-dim);
        margin-bottom: 8px;
    }
    .signal-value {
        font-size: 26px;
        font-weight: 800;
        line-height: 1.1;
        color: var(--ink);
        margin-bottom: 8px;
        font-variant-numeric: tabular-nums;
    }
    .signal-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 11px;
        font-weight: 700;
    }
    .signal-pill.up { background: rgba(232,66,110,.12); color: var(--pink-deep); }
    .signal-pill.down { background: rgba(61,184,158,.15); color: #1a8c74; }
    .signal-pill.warn { background: rgba(232,66,110,.15); color: var(--pink-deep); }
    .signal-help {
        font-size: 11px;
        color: var(--ink-dim);
        line-height: 1.4;
        margin-top: 8px;
    }

    /* ── 출연진 카드 ── */
    .cast-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 4px 0 18px;
    }
    .cast-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        overflow: hidden;
        transition: transform .25s, border-color .2s, box-shadow .2s;
        cursor: default;
    }
    .cast-card:hover {
        transform: translateY(-4px);
        border-color: var(--line);
        box-shadow: var(--glow-pink);
    }
    .cast-photo {
        width: 100%;
        height: 120px;
        object-fit: cover;
        display: block;
        filter: saturate(.75) brightness(.85);
    }
    .cast-photo-placeholder {
        width: 100%;
        height: 130px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 52px;
        background: linear-gradient(160deg, rgba(232,66,110,.10), rgba(232,66,110,.04));
        border-bottom: 1px solid var(--line-soft);
        gap: 4px;
    }
    .cast-body { padding: 11px 12px 13px; }
    .cast-ticker {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .cast-name {
        font-size: 16px;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 4px;
        line-height: 1.25;
    }
    .cast-role {
        font-size: 11px;
        color: var(--ink-dim);
        margin-bottom: 6px;
        font-weight: 500;
    }
    .cast-tagline {
        font-size: 11px;
        color: var(--ink-muted);
        line-height: 1.45;
    }

    /* ── 패널 노트 ── */
    .note-panel {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 18px;
        backdrop-filter: blur(10px);
    }
    .note-kicker {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--pink);
        margin-bottom: 8px;
    }
    .note-title {
        font-family: "Nanum Myeongjo", serif;
        font-size: 20px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 12px;
        line-height: 1.3;
    }
    .note-line {
        border-left: 2px solid var(--pink);
        padding: 6px 0 6px 12px;
        margin: 8px 0;
        font-size: 13px;
        line-height: 1.55;
        color: var(--ink);
    }

    /* ── 배분 슬라이더 폼 ── */
    .predict-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 18px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }
    .predict-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
    }
    .predict-badge {
        background: linear-gradient(135deg, var(--pink-deep), var(--pink));
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .06em;
        text-transform: uppercase;
        color: #fff;
    }

    /* ── AI 잠금 패널 ── */
    .locked-panel {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 28px;
        text-align: center;
        backdrop-filter: blur(10px);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .locked-icon { font-size: 36px; margin-bottom: 10px; }
    .locked-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 6px;
    }
    .locked-sub {
        font-size: 12px;
        color: var(--ink-dim);
        line-height: 1.5;
        max-width: 220px;
    }

    /* ── 결과 카드 ── */
    .verdict-card {
        border-radius: var(--radius);
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 14px;
    }
    .verdict-win {
        background: rgba(61,184,158,.12);
        border: 1px solid rgba(61,184,158,.40);
        border-left: 4px solid var(--teal);
    }
    .verdict-lose {
        background: rgba(232,66,110,.10);
        border: 1px solid rgba(232,66,110,.35);
        border-left: 4px solid var(--pink);
    }
    .verdict-kicker {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--ink-dim);
        margin-bottom: 6px;
    }
    .verdict-title {
        font-family: "Nanum Myeongjo", serif;
        font-size: 22px;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 12px;
    }

    /* ── 상금 카드 ── */
    .prize-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 14px 0;
    }
    .prize-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .prize-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
        color: var(--ink-dim);
        margin-bottom: 8px;
    }
    .prize-amount {
        font-size: 26px;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 6px;
        font-variant-numeric: tabular-nums;
    }
    .prize-sub {
        font-size: 11px;
        color: var(--ink-dim);
        line-height: 1.4;
    }
    .pos { color: var(--teal); }
    .neg { color: var(--pink); }

    /* ── 국면 가이드 카드 ── */
    .regime-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin: 12px 0;
    }
    .regime-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    .regime-img {
        width: 100%;
        height: 90px;
        object-fit: cover;
        filter: brightness(.5) saturate(.7);
        display: block;
    }
    .regime-body { padding: 14px; }
    .regime-ep-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .05em;
        margin-bottom: 8px;
        color: #fff;
    }
    .regime-name {
        font-family: "Nanum Myeongjo", serif;
        font-size: 16px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .mini-weight {
        display: grid;
        grid-template-columns: 36px 1fr 34px;
        align-items: center;
        gap: 7px;
        margin-top: 6px;
        font-size: 11px;
        color: var(--ink-muted);
        font-variant-numeric: tabular-nums;
    }
    .mini-track {
        height: 5px;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        overflow: hidden;
    }
    .mini-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--pink), var(--teal));
    }

    /* ── 엔트리 스크린 ── */
    .entry-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        min-height: 340px;
        display: flex;
        align-items: center;
        box-shadow: var(--shadow), var(--glow-pink);
        margin-bottom: 20px;
    }
    .entry-hero-bg {
        position: absolute; inset: 0;
        background-size: cover;
        background-position: center 30%;
        filter: brightness(.3) saturate(.7);
    }
    .entry-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(135deg,
            rgba(255,240,247,.92) 0%,
            rgba(255,230,242,.72) 50%,
            rgba(255,220,238,.52) 100%);
    }
    .entry-inner {
        position: relative; z-index: 2;
        padding: 40px 48px;
        max-width: 780px;
    }
    .entry-kicker {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .10em;
        text-transform: uppercase;
        color: var(--pink-deep);
        margin-bottom: 14px;
    }
    .entry-title {
        font-family: "Nanum Myeongjo", serif;
        font-size: 52px;
        font-weight: 800;
        line-height: 1.10;
        color: var(--ink);
        margin-bottom: 14px;
        text-shadow: 0 1px 12px rgba(200,80,120,.18);
    }
    .entry-sub {
        font-size: 15px;
        line-height: 1.65;
        color: var(--ink-muted);
        margin-bottom: 24px;
        max-width: 560px;
    }
    .entry-vs {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 20px;
    }
    .entry-vs-pill {
        background: rgba(255,255,255,.85);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 7px 18px;
        font-size: 13px;
        font-weight: 700;
        color: var(--ink);
        backdrop-filter: blur(8px);
        box-shadow: 0 2px 12px rgba(200,80,120,.10);
    }
    .entry-vs-sep { color: var(--pink); font-size: 20px; font-weight: 900; }

    /* ── 모드 버튼 (entry screen) ── */
    .mode-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 16px;
    }
    .mode-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: border-color .2s, background .2s;
    }
    .mode-card:hover {
        background: var(--panel-hover);
        border-color: var(--line);
    }
    .mode-tag {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .mode-name {
        font-size: 20px;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 6px;
    }
    .mode-desc {
        font-size: 12px;
        color: var(--ink-dim);
        line-height: 1.5;
    }

    /* ── 가이드 탭 섹션 ── */
    .guide-section {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 22px;
        backdrop-filter: blur(10px);
        margin-bottom: 14px;
    }
    .guide-kicker {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--pink);
        margin-bottom: 8px;
    }
    .guide-title {
        font-family: "Nanum Myeongjo", serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 10px;
        line-height: 1.3;
    }
    .guide-sub {
        font-size: 13px;
        color: var(--ink-muted);
        line-height: 1.6;
    }

    /* ── 흐름 카드 ── */
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 12px 0;
    }
    .flow-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: var(--radius);
        padding: 14px;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .flow-num {
        width: 28px; height: 28px;
        border-radius: 999px;
        background: rgba(240,96,138,.18);
        color: var(--pink-light);
        font-weight: 900;
        font-size: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px;
    }
    .flow-label {
        font-size: 13px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 4px;
    }
    .flow-desc {
        font-size: 11px;
        color: var(--ink-dim);
        line-height: 1.45;
    }

    /* ── 엔딩 타이틀 ── */
    .ending-hero {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 28px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
        box-shadow: var(--glow-pink);
    }
    .ending-title {
        font-family: "Nanum Myeongjo", serif;
        font-size: 36px;
        font-weight: 800;
        color: var(--pink-light);
        margin-bottom: 8px;
    }
    .ending-sub {
        font-size: 14px;
        color: var(--ink-muted);
        line-height: 1.65;
        max-width: 560px;
        margin: 0 auto;
    }

    /* ── 스트림릿 기본 오버라이드 ── */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(240,96,138,.5) !important;
        background: linear-gradient(135deg, var(--pink-deep), var(--pink)) !important;
        color: #fff !important;
        font-weight: 800 !important;
        font-family: "Pretendard", "Noto Sans KR", sans-serif !important;
        box-shadow: 0 8px 24px rgba(192,48,96,.28) !important;
        padding: 0.5rem 1.4rem !important;
        transition: transform .15s, box-shadow .15s !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(192,48,96,.40) !important;
    }
    div[data-testid="stMetric"] {
        background: var(--panel) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: var(--radius) !important;
        padding: 12px !important;
        backdrop-filter: blur(8px) !important;
    }
    div[data-testid="stMetric"] label { color: var(--ink-dim) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--teal) !important; }
    .stDataFrame { background: transparent !important; }
    .stSelectbox > div > div {
        background: var(--panel) !important;
        border-color: var(--line-soft) !important;
        color: var(--ink) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stSlider > div { color: var(--ink) !important; }
    .stProgress > div { border-radius: 999px !important; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { color: var(--ink-muted) !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--pink-light) !important; }
    .stTabs [data-baseweb="tab-border"] { background: var(--line) !important; }

    @media (max-width: 1100px) {
        .signal-grid, .cast-grid { grid-template-columns: repeat(2, 1fr); }
        .flow-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-inner { grid-template-columns: 1fr; }
        .hero-photo-wrap { display: none; }
        .entry-title { font-size: 36px; }
        .entry-inner { padding: 28px 24px; }
        .regime-grid { grid-template-columns: 1fr; }
        .mode-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_game_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True).sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.apply(pd.to_numeric, errors="coerce").ffill().bfill()

    if HMM_DATA_PATH.exists():
        hmm = pd.read_csv(HMM_DATA_PATH, index_col=0, parse_dates=True).sort_index()
        hmm.index = pd.to_datetime(hmm.index).tz_localize(None).normalize()
        hmm_cols = ["Level", "Equity_vs_Bond", "Jobless_Claims_MA", "Market_Breadth", "spread_10y2y", "Copper_Gold_Ratio"]
        df = raw.join(hmm[hmm_cols], how="left")
    else:
        df = raw.copy()
        df["Level"] = pd.qcut(df["VIX"].rank(method="first"), 6, labels=[2, 1, 3, 6, 4, 5]).astype(int)
        df["Equity_vs_Bond"] = df["SPY"] / df["TLT"]
        df["Jobless_Claims_MA"] = df["Jobless_Claims"].rolling(LOOKBACK_DAYS).mean()
        df["Market_Breadth"] = df["RSP"] / df["SPY"]
        df["spread_10y2y"] = df["US10Y"] - df["US2Y"]
        df["Copper_Gold_Ratio"] = df["Copper"] / df["Gold"]

    df["SPY_Mom_1M"] = df["SPY"].pct_change(LOOKBACK_DAYS)
    df["VIX_Change_1M"] = df["VIX"].diff(LOOKBACK_DAYS)
    df["Claims_Change_1M"] = df["Jobless_Claims_MA"].diff(LOOKBACK_DAYS)
    df["Copper_Gold_Change_1M"] = df["Copper_Gold_Ratio"].pct_change(LOOKBACK_DAYS)

    keep = ASSETS + [
        "VIX", "DXY", "US10Y", "US2Y", "US3M", "Jobless_Claims",
        "Level", "Equity_vs_Bond", "Jobless_Claims_MA", "Market_Breadth",
        "spread_10y2y", "Copper_Gold_Ratio",
        "SPY_Mom_1M", "VIX_Change_1M", "Claims_Change_1M", "Copper_Gold_Change_1M",
    ]
    keep = [col for col in dict.fromkeys(keep) if col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()][keep]
    return df.dropna(subset=["Level", "VIX", "spread_10y2y", "SPY", "Copper_Gold_Ratio", "Jobless_Claims_MA"])


# ══════════════════════════════════════════════════════════════════════════════
# GAME LOGIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def valid_scenario_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.index[LOOKBACK_DAYS + 1 : len(df) - HORIZON_DAYS - 1]


def pick_random_date(df: pd.DataFrame) -> pd.Timestamp:
    dates = valid_scenario_dates(df)
    rng = np.random.default_rng()
    return pd.Timestamp(rng.choice(dates.to_numpy()))


def normalize(weights, assets: list[str]) -> pd.Series:
    if isinstance(weights, dict):
        series = pd.Series(weights, dtype=float)
    elif isinstance(weights, pd.Series):
        series = weights.astype(float)
    else:
        series = pd.Series(weights, index=assets, dtype=float)
    series = series.reindex(assets).fillna(0).clip(lower=0)
    total = float(series.sum())
    if total <= 0:
        return pd.Series(np.ones(len(assets)) / len(assets), index=assets)
    return series / total


def recommended_weights(level: int, assets: list[str]) -> pd.Series:
    return normalize(RECOMMENDED_WEIGHTS[level], assets)


def infer_signal_regime(row: pd.Series) -> int:
    vix = float(row["VIX"])
    spy_mom = float(row["SPY_Mom_1M"])
    vix_change = float(row["VIX_Change_1M"])
    spread = float(row["spread_10y2y"])
    claims_change = float(row["Claims_Change_1M"])
    copper_gold_change = float(row["Copper_Gold_Change_1M"])

    stress = 0
    stress += 3 if vix >= 32 else 2 if vix >= 24 else 1 if vix >= 18 else 0
    stress += 3 if spy_mom <= -0.09 else 2 if spy_mom <= -0.04 else 1 if spy_mom < 0 else 0
    stress += 2 if vix_change >= 6 else 1 if vix_change >= 2 else 0
    stress += 1 if spread < 0 else 0
    stress += 1 if claims_change >= 20000 else 0
    stress += 1 if copper_gold_change <= -0.04 else 0

    fear_easing = vix_change <= -4 and spy_mom > -0.03
    strong_risk_on = vix <= 18 and spy_mom >= 0.025 and spread >= 0 and claims_change < 20000
    overheated = vix <= 20 and spy_mom >= 0.065
    quiet_repair = vix <= 22 and -0.035 <= spy_mom <= 0.035 and vix_change <= 2

    if stress >= 7 or (vix >= 30 and spy_mom <= -0.04):
        return 5
    if fear_easing and (vix >= 18 or spy_mom > 0):
        return 6
    if stress >= 4:
        return 4
    if overheated:
        return 3
    if strong_risk_on:
        return 2
    if quiet_repair:
        return 1
    return 4 if stress >= 2 else 1


def ai_panel_weights(row: pd.Series, assets: list[str], level: int) -> pd.Series:
    weights = recommended_weights(level, assets).copy()
    equity_assets = [a for a in ["SPY"] if a in assets]
    bond_assets = [a for a in ["TLT", "SHV"] if a in assets]

    def shift(src, dst, amount):
        if not src or not dst or amount <= 0:
            return
        take = min(amount, float(weights[src].sum()) * 0.65)
        if take <= 0:
            return
        for a in src:
            weights[a] -= take * (weights[a] / weights[src].sum())
        for a in dst:
            weights[a] += take / len(dst)

    if row["VIX"] > 25 or row["VIX_Change_1M"] > 4:
        shift(equity_assets + ["DBC"], [a for a in ["SHV", "TLT", "GLD"] if a in assets], 0.07)
    if row["spread_10y2y"] < 0:
        shift(equity_assets, bond_assets, 0.05)
    if row["SPY_Mom_1M"] > 0.05 and row["VIX"] < 18 and level in [1, 2, 6]:
        shift([a for a in ["SHV", "TLT"] if a in assets], equity_assets, 0.06)
    if row["Copper_Gold_Change_1M"] > 0.04 and "DBC" in assets:
        shift([a for a in ["SHV"] if a in assets], ["DBC"], 0.03)
    if row["Claims_Change_1M"] > 25000:
        shift(equity_assets, [a for a in ["SHV", "GLD"] if a in assets], 0.04)

    return normalize(weights, assets)


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def plain_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_krw(value) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}{abs(int(round(value))):,}원"


def format_delta(value: float, suffix: str = "") -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.2f}{suffix}"


def regime_option(level: int) -> str:
    r = REGIMES[level]
    return f"{r['episode']} / {r['finance']}"


def one_month_window(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    loc = df.index.get_loc(date)
    return df.iloc[loc - LOOKBACK_DAYS : loc + 1]


def portfolio_path(df, date, horizon, weights, assets):
    start_loc = df.index.get_loc(date)
    window = df.iloc[start_loc : start_loc + horizon + 1][assets]
    rets = window.pct_change().dropna()
    port_rets = rets.dot(weights.reindex(assets))
    values = (1 + port_rets).cumprod()
    return pd.concat([pd.Series([1.0], index=[window.index[0]]), values])


def asset_forward_returns(df, date, horizon, assets):
    loc = df.index.get_loc(date)
    start = df.iloc[loc][assets]
    end = df.iloc[loc + horizon][assets]
    return end / start - 1


def metrics(path):
    rets = path.pct_change().dropna()
    total_return = float(path.iloc[-1] / path.iloc[0] - 1)
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    mdd = float((path / path.cummax() - 1).min())
    return {"total_return": total_return, "vol": vol, "mdd": mdd}


def allocation_fit_score(weights, target):
    dist = float((weights.reindex(target.index).fillna(0) - target).abs().sum())
    return float(np.clip(100 * (1 - dist / 2), 0, 100))


def regime_guess_score(actual, guess):
    d = abs(actual - guess)
    if d == 0: return 100.0
    if d == 1: return 75.0
    if d == 2: return 55.0
    return 35.0


def round_score(m, signal_score):
    raw = 46 + m["total_return"] * 650 + signal_score * 0.42 + m["mdd"] * 250 - m["vol"] * 18
    return float(np.clip(raw, 0, 100))


def panel_type(weights):
    equity = float(weights.reindex(["SPY"]).fillna(0).sum())
    defense = float(weights.reindex(["TLT", "SHV"]).fillna(0).sum())
    hedge = float(weights.reindex(["GLD", "DBC"]).fillna(0).sum())
    if weights.get("SHV", 0) >= 0.42: return "🏠 안전주의 패널"
    if equity >= 0.58: return "💘 직진형 패널"
    if defense >= 0.58: return "🫶 안정형 패널"
    if hedge >= 0.36: return "✨ 위기감지형 패널"
    return "⚖️ 밸런스형 패널"


def observation_notes(row):
    notes = []
    if row["VIX_Change_1M"] > 3:
        notes.append("VIX가 1개월 전보다 뚜렷하게 올라 시장의 감정 기복이 커졌습니다.")
    elif row["VIX_Change_1M"] < -3:
        notes.append("VIX가 1개월 전보다 내려오며 공포가 완화되는 흐름입니다.")
    else:
        notes.append("VIX는 큰 폭의 변화 없이 관찰 구간 안에서 움직이고 있습니다.")
    if row["spread_10y2y"] < 0:
        notes.append("10년물과 2년물 금리차가 역전되어 경기 둔화 경계감이 남아 있습니다.")
    else:
        notes.append("10년물과 2년물 금리차는 플러스권으로 경기 신호가 완전히 꺾인 모습은 아닙니다.")
    if row["SPY_Mom_1M"] > 0.04:
        notes.append("SPY의 1개월 모멘텀이 강해 위험자산 쪽으로 시선이 붙고 있습니다.")
    elif row["SPY_Mom_1M"] < -0.04:
        notes.append("SPY가 1개월 기준 밀리며 공격형 출연진의 매력이 약해졌습니다.")
    else:
        notes.append("SPY는 뚜렷한 방향성보다 탐색전에 가까운 흐름입니다.")
    if row["Claims_Change_1M"] > 20000:
        notes.append("실업수당 청구건수 이동평균이 올라 고용 부담을 확인해야 합니다.")
    elif row["Claims_Change_1M"] < -20000:
        notes.append("실업수당 청구건수 이동평균이 낮아져 고용 불안은 다소 완화됐습니다.")
    return notes


def result_review(actual_level, guess, user_w, ai_w, user_return, ai_return):
    regime = REGIMES[actual_level]
    guessed = REGIMES[guess]["finance"]
    defense_user = float(user_w.reindex(["TLT", "SHV", "GLD"]).fillna(0).sum())
    equity_user = float(user_w.reindex(["SPY"]).fillna(0).sum())

    if guess == actual_level:
        first = f"국면 예측은 정확했습니다. 실제 회차는 {regime['episode']}였습니다."
    else:
        first = f"플레이어는 {guessed}를 예상했지만 실제 회차는 {regime['episode']}였습니다."

    second = ("수익률 대결에서 플레이어 패널이 AI 패널 P보다 시장의 마음을 더 잘 읽었습니다."
              if user_return >= ai_return
              else "수익률 대결에서 AI 패널 P가 더 안정적인 선택을 했습니다.")

    if actual_level in [4, 5] and defense_user < 0.55:
        third = "불안 신호가 강한 회차에서는 방어형 출연진 비중을 더 키우면 오답 충격도를 줄일 수 있습니다."
    elif actual_level in [2, 6] and equity_user < 0.40:
        third = "위험자산으로 시선이 옮겨가는 회차에서는 SPY 비중을 너무 낮게 두면 반등 기회를 놓칩니다."
    else:
        third = f"현재 배분은 {panel_type(user_w)} 성향으로 이번 국면의 핵심 시그널을 일부 반영했습니다."
    return f"{first} {second} {third}"


def weight_bars_html(level):
    rows = []
    weights = recommended_weights(level, ASSETS)
    for asset in ASSETS:
        pct = int(round(weights[asset] * 100))
        color = ASSET_META[asset]["color"]
        rows.append(
            f'<div class="mini-weight">'
            f'<b style="color:{color}">{asset}</b>'
            f'<div class="mini-track"><div class="mini-fill" style="width:{pct}%;background:{color}88"></div></div>'
            f'<span>{pct}%</span>'
            f'</div>'
        )
    return "".join(rows)


def next_episode_date(current, df):
    options = valid_scenario_dates(df)[::LOOKBACK_DAYS]
    pos = int(np.searchsorted(options.to_numpy(), np.datetime64(current), side="right"))
    if pos >= len(options): pos = 0
    return pd.Timestamp(options[pos])


def ending_title(history):
    player_cum = float(np.prod([1 + float(i["user_return"]) for i in history]) - 1)
    ai_cum = float(np.prod([1 + float(i["ai_return"]) for i in history]) - 1)
    avg_signal = float(np.mean([float(i["signal_score"]) for i in history]))
    crisis_survived = any(int(i["level"]) == 5 and float(i["user_return"]) >= float(i["ai_return"]) for i in history)
    expansion_best = any(int(i["level"]) == 2 and float(i["user_return"]) > 0.03 for i in history)
    euphoria_defense = any(int(i["level"]) == 3 and float(i["mdd"]) > -0.02 for i in history)
    bottoming_switch = any(int(i["level"]) == 6 and float(i["user_return"]) >= float(i["ai_return"]) for i in history)

    if player_cum > ai_cum: return "🏆 AI를 이긴 인간 패널"
    if avg_signal >= 85: return "🔍 시그널 해석 장인"
    if crisis_survived: return "🚨 대혼란 회차 생존 패널"
    if expansion_best: return "💘 직진 예측 성공 패널"
    if euphoria_defense: return "🛡 과몰입 방지 패널"
    if bottoming_switch: return "🌅 새로운 시그널 포착 패널"
    if player_cum > -0.02: return "🔭 신중한 관찰형 패널"
    return "😵 과몰입 패널"


@st.cache_data(show_spinner=False)
def studio_image_src() -> str:
    for path in STUDIO_IMAGE_CANDIDATES:
        if path.exists():
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return STUDIO_IMAGE_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def make_indicator_chart(window):
    cols = {"VIX": "VIX", "spread_10y2y": "10Y-2Y", "SPY": "SPY", "Copper_Gold_Ratio": "Copper/Gold", "Jobless_Claims_MA": "Jobless MA"}
    palette = ["#e8426e", "#5a8fc4", "#c2245a", "#3db89e", "#c9873a"]
    fig = go.Figure()
    for color, (col, label) in zip(palette, cols.items()):
        series = window[col].dropna()
        start = float(series.iloc[0])
        normalized = 100 + (series - start) * 20 if abs(start) < 1 else series / start * 100
        fig.add_trace(go.Scatter(x=series.index, y=normalized, mode="lines", name=label,
                                 line=dict(color=color, width=2.5)))
    fig.update_layout(
        template=None, height=300, margin=dict(l=28, r=18, t=48, b=34),
        paper_bgcolor="rgba(255,245,247,0)", plot_bgcolor="rgba(255,245,247,0)",
        yaxis=dict(title="선택일 1개월 전 = 100", gridcolor="rgba(200,80,120,.10)",
                   tickfont=dict(color="rgba(58,26,40,.50)", size=11), color="rgba(58,26,40,.50)"),
        xaxis=dict(gridcolor="rgba(200,80,120,.07)",
                   tickfont=dict(color="rgba(58,26,40,.40)", size=10), color="rgba(58,26,40,.40)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
                    font=dict(size=11, color="rgba(58,26,40,.65)"),
                    bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Pretendard, Noto Sans KR, sans-serif"),
    )
    return fig


def make_weight_chart(user_w, ai_w, assets):
    colors = [ASSET_META[a]["color"] for a in assets]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=assets, y=user_w.reindex(assets).values * 100,
                         name="🩷 플레이어", marker_color=colors,
                         marker=dict(line=dict(width=0)), opacity=.90))
    fig.add_trace(go.Bar(x=assets, y=ai_w.reindex(assets).values * 100,
                         name="🤖 AI 패널 P", marker_color="rgba(180,140,160,.45)",
                         marker=dict(line=dict(width=0))))
    fig.update_layout(
        template=None, barmode="group", height=290, margin=dict(l=28, r=18, t=48, b=34),
        paper_bgcolor="rgba(255,245,247,0)", plot_bgcolor="rgba(255,245,247,0)",
        yaxis=dict(title="비중 (%)", gridcolor="rgba(200,80,120,.10)",
                   tickfont=dict(color="rgba(58,26,40,.50)", size=11), color="rgba(58,26,40,.50)"),
        xaxis=dict(tickfont=dict(color="rgba(58,26,40,.65)", size=12), color="rgba(58,26,40,.50)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
                    font=dict(size=11, color="rgba(58,26,40,.65)"), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Pretendard, Noto Sans KR, sans-serif"),
    )
    return fig


def make_result_chart(user_path, ai_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=user_path.index, y=user_path.values, mode="lines",
                             name="🩷 플레이어", line=dict(color="#e8426e", width=3)))
    fig.add_trace(go.Scatter(x=ai_path.index, y=ai_path.values, mode="lines",
                             name="🤖 AI 패널 P", line=dict(color="#3db89e", width=3)))
    fig.update_layout(
        template=None, height=290, margin=dict(l=28, r=18, t=48, b=34),
        paper_bgcolor="rgba(255,245,247,0)", plot_bgcolor="rgba(255,245,247,0)",
        yaxis=dict(title="자산 가치", gridcolor="rgba(200,80,120,.10)",
                   tickfont=dict(color="rgba(58,26,40,.50)", size=11), color="rgba(58,26,40,.50)"),
        xaxis=dict(gridcolor="rgba(200,80,120,.07)",
                   tickfont=dict(color="rgba(58,26,40,.40)", size=10), color="rgba(58,26,40,.40)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
                    font=dict(size=11, color="rgba(58,26,40,.65)"), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Pretendard, Noto Sans KR, sans-serif"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_signal_cards(row, window):
    spy_1m = float(row["SPY_Mom_1M"])
    vix_delta = float(row["VIX_Change_1M"])
    spread_delta = float(row["spread_10y2y"] - window["spread_10y2y"].iloc[0])
    cg_delta = float(row["Copper_Gold_Change_1M"])
    claims_delta = float(row["Claims_Change_1M"])

    cards = [
        ("VIX", f"{row['VIX']:.1f}", f"1개월 {format_delta(vix_delta)}", "상승하면 불안과 변동성이 커집니다.", vix_delta > 0),
        ("10Y-2Y 금리차", f"{row['spread_10y2y']:.2f}", f"1개월 {format_delta(spread_delta, 'p')}", "음수면 경기 둔화 경계 신호입니다.", row["spread_10y2y"] < 0),
        ("SPY 가격", f"{row['SPY']:.2f}", f"1개월 {format_pct(spy_1m)}", "1개월 모멘텀이 시장의 시선을 보여줍니다.", spy_1m < 0),
        ("Copper/Gold", f"{row['Copper_Gold_Ratio']:.4f}", f"1개월 {format_pct(cg_delta)}", "오르면 경기민감 선호가 강해집니다.", cg_delta < 0),
        ("Jobless MA", f"{row['Jobless_Claims_MA'] / 1000:.0f}K", f"1개월 {format_delta(claims_delta / 1000, 'K')}", "오르면 고용 둔화 부담이 커집니다.", claims_delta > 0),
    ]

    html = ['<div class="signal-grid">']
    for label, value, delta, help_text, warn in cards:
        card_cls = "signal-card " + ("warn" if warn else "safe")
        pill_cls = "signal-pill " + ("warn" if warn else "down")
        arrow = "▲" if "+" in delta and warn else ("▲" if "+" in delta else "▼")
        html.append(
            f'<div class="{card_cls}">'
            f'<div class="signal-label">{label}</div>'
            f'<div class="signal-value">{value}</div>'
            f'<div class="{pill_cls}">{arrow} {delta}</div>'
            f'<div class="signal-help">{help_text}</div>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_cast_cards(assets):
    html = ['<div class="cast-grid">']
    for asset in assets:
        meta = ASSET_META[asset]
        photo_url = ASSET_PHOTOS.get(asset, "")
        if photo_url:
            photo_html = f'<img class="cast-photo" src="{photo_url}" alt="{meta["name"]}" onerror="this.style.display=\'none\'">'
        else:
            photo_html = f'<div class="cast-photo-placeholder">{meta["emoji"]}</div>'
        html.append(
            f'<div class="cast-card">'
            f'{photo_html}'
            f'<div class="cast-body">'
            f'<div class="cast-ticker" style="color:{meta["color"]}">{asset} {meta["emoji"]}</div>'
            f'<div class="cast-name">{meta["name"]}</div>'
            f'<div class="cast-role">{meta["role"]}</div>'
            f'<div class="cast-tagline">{meta["tagline"]}</div>'
            f'</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def render_entry_screen():
    studio_src = studio_image_src()
    st.markdown(
        f"""
        <div class="entry-hero">
            <div class="entry-hero-bg" style="background-image:url('{studio_src}')"></div>
            <div class="entry-overlay"></div>
            <div class="entry-inner">
                <div class="entry-kicker">💘 Signal House — Season 1</div>
                <div class="entry-title">마켓시그널<br>AI 패널을 이겨라</div>
                <div class="entry-sub">
                    시장의 마음은 오늘 어떤 ETF에게 향할까?<br>
                    경제 시그널을 읽고 ETF 출연진의 러브라인을 예측하세요.<br>
                    PPO 기반 AI 패널 P보다 더 높은 적중률을 기록할 수 있을까요?
                </div>
                <div class="entry-vs">
                    <div class="entry-vs-pill">🩷 인간 패널 (나)</div>
                    <div class="entry-vs-sep">VS</div>
                    <div class="entry-vs-pill">🤖 AI 패널 P</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 출연진 소개 카드 (5개 ETF 단일 구성)
    st.markdown(
        """
        <div class="mode-grid" style="grid-template-columns:1fr">
            <div class="mode-card" style="border-color:rgba(232,66,110,.40);background:rgba(255,255,255,.80)">
                <div class="mode-tag" style="color:var(--pink)">💘 SIGNAL HOUSE — Season 1</div>
                <div class="mode-name">오늘의 출연진 5인</div>
                <div class="mode-desc">
                    <b style="color:#e8426e">SPY</b> 직진남 &nbsp;·&nbsp;
                    <b style="color:#4f7fb8">TLT</b> 돈많은 연하남 &nbsp;·&nbsp;
                    <b style="color:#687385">SHV</b> 무해한 집돌이 &nbsp;·&nbsp;
                    <b style="color:#c69b3f">GLD</b> 신비주의 연상남 &nbsp;·&nbsp;
                    <b style="color:#3db89e">DBC</b> 핫한 자유영혼<br><br>
                    경제 시그널을 연애 감정처럼 읽고, 시장의 마음이 향하는 출연진에게 비중을 배분하세요.<br>
                    PPO 기반 AI 패널 P를 이기면 시그널 원석과 상금을 획득합니다.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("역할", "🩷 인간 패널")
    c2.metric("상대", "🤖 AI 패널 P")
    c3.metric("총 회차", "6회차")
    c4.metric("회차 상금", format_krw(SIGNAL_GEM_VALUE))

    st.write("")
    if st.button("💘 패널석 입장하기", use_container_width=True):
        st.session_state.entered = True
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# GAME GUIDE
# ══════════════════════════════════════════════════════════════════════════════
def render_game_guide():
    st.markdown("---")
    st.markdown(
        """
        <div class="guide-section">
            <div class="guide-kicker">Signal House Guide</div>
            <div class="guide-title">경제 지표를 연애 시그널처럼 읽는 자산배분 게임</div>
            <div class="guide-sub">
                사용자는 투자자가 아니라 <b>인간 패널</b>입니다.<br>
                날짜를 선택하면 1개월치 국면 힌트가 공개되고, 그 힌트만 보고 시장의 마음이 어떤 ETF 출연자에게 향할지 비중으로 예측합니다.<br>
                PPO 기반 AI 패널 P와 다음 회차 수익률로 대결하여 시그널 원석과 상금을 쌓으세요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rule_tab, cast_tab, regime_tab, panel_tab, ending_tab = st.tabs(
        ["📋 게임 룰", "💘 출연진", "📺 국면 가이드", "🤖 AI·점수", "🏆 엔딩"]
    )

    with rule_tab:
        st.markdown(
            """
            <div class="flow-grid">
                <div class="flow-card">
                    <div class="flow-num">1</div>
                    <div class="flow-label">날짜 선택</div>
                    <div class="flow-desc">관찰 날짜를 고르면 직전 1개월 시장 시그널이 공개됩니다.</div>
                </div>
                <div class="flow-card">
                    <div class="flow-num">2</div>
                    <div class="flow-label">국면 추측</div>
                    <div class="flow-desc">VIX, 금리차, SPY, 구리/금, 실업수당 지표만 보고 현재 국면을 맞힙니다.</div>
                </div>
                <div class="flow-card">
                    <div class="flow-num">3</div>
                    <div class="flow-label">마음 배분</div>
                    <div class="flow-desc">5명의 ETF 출연진에게 시장의 마음을 비중으로 나눕니다.</div>
                </div>
                <div class="flow-card">
                    <div class="flow-num">4</div>
                    <div class="flow-label">AI와 승부</div>
                    <div class="flow-desc">다음 1개월 수익률로 AI 패널 P와 회차 점수를 비교합니다.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="prize-grid">
                <div class="prize-card">
                    <div class="prize-label">승리 시</div>
                    <div class="prize-amount pos">+1 원석</div>
                    <div class="prize-sub">AI보다 회차 점수가 높으면 {format_krw(SIGNAL_GEM_VALUE)}이 적립됩니다.</div>
                </div>
                <div class="prize-card">
                    <div class="prize-label">패배 시</div>
                    <div class="prize-amount neg">−1 원석</div>
                    <div class="prize-sub">AI에게 지면 {format_krw(-SIGNAL_GEM_VALUE)}이 차감됩니다.</div>
                </div>
                <div class="prize-card">
                    <div class="prize-label">최종 엔딩</div>
                    <div class="prize-amount">6회차</div>
                    <div class="prize-sub">누적 수익률·시그널 해석력·최종 상금판으로 결과가 공개됩니다.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cast_tab:
        st.markdown("#### 시그널 하우스 출연진 소개")
        cast_guide = ['<div class="cast-grid">']
        for asset in ASSETS:
            meta = ASSET_META[asset]
            photo_url = ASSET_PHOTOS.get(asset, "")
            photo_html = (f'<img class="cast-photo" src="{photo_url}" alt="{meta["name"]}">'
                          if photo_url else f'<div class="cast-photo-placeholder">{meta["emoji"]}</div>')
            cast_guide.append(
                f'<div class="cast-card">'
                f'{photo_html}'
                f'<div class="cast-body">'
                f'<div class="cast-ticker" style="color:{meta["color"]}">{asset} {meta["emoji"]}</div>'
                f'<div class="cast-name">{meta["name"]}</div>'
                f'<div class="cast-role">{meta["role"]}</div>'
                f'<div class="cast-tagline">{meta["tagline"]}</div>'
                f'<div class="cast-role" style="margin-top:6px;color:var(--ink-muted)">{meta["profile"]}</div>'
                f'</div></div>'
            )
        cast_guide.append("</div>")
        st.markdown("".join(cast_guide), unsafe_allow_html=True)

    with regime_tab:
        st.markdown("#### 6개 회차 국면 가이드")
        regime_cards = ['<div class="regime-grid">']
        for level, regime in REGIMES.items():
            photo = REGIME_PHOTOS.get(level, STUDIO_IMAGE_FALLBACK)
            ep_alt = regime["episode"]
            regime_cards.append(
                f'<div class="regime-card">'
                f'<img class="regime-img" src="{photo}" alt="{ep_alt}">'
                f'<div class="regime-body">'
                f'<div class="regime-ep-badge" style="background:{regime["badge_color"]}">EP.{level:02d} · {regime["finance"]}</div>'
                f'<div class="regime-name">{regime["episode"]}</div>'
                f'<div class="note-line">{regime["mood"]}<br><b>시장 시그널</b>: {regime["signal"]}</div>'
                f'<div class="note-line"><b>패널 해석</b>: {regime["panel"]}</div>'
                f'<div class="note-line"><b>배분 힌트</b>: {regime["guide"]}</div>'
                f'<div style="font-size:11px;color:var(--ink-dim);margin-top:8px;font-weight:700;text-transform:uppercase;letter-spacing:.05em">참고 비중</div>'
                f'{weight_bars_html(level)}'
                f'</div></div>'
            )
        regime_cards.append("</div>")
        st.markdown("".join(regime_cards), unsafe_allow_html=True)

    with panel_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown(
                """
                <div class="note-panel">
                    <div class="note-kicker">AI Panel P</div>
                    <div class="note-title">🤖 AI 패널 P란?</div>
                    <div class="note-line">플레이어와 같은 1개월 시그널을 보고 5개 ETF 비중을 정합니다. AI의 선택은 플레이어가 예측을 제출한 뒤 공개됩니다.</div>
                    <div class="note-line">목표는 AI보다 다음 1개월 시장의 마음을 더 잘 읽는 것입니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                """
                <div class="note-panel">
                    <div class="note-kicker">Round Score</div>
                    <div class="note-title">📊 회차 점수 계산</div>
                    <div class="note-line">수익률 + 국면 예측 + 추천 배분과의 거리 + 오답 충격도를 종합 반영합니다.</div>
                    <div class="note-line">플레이어 점수 > AI 점수 → 시그널 원석 1개 획득 & 상금 +50만원</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="verdict-card verdict-lose" style="margin-top:12px">
                <div class="verdict-kicker">EP.04 엇갈리는 문자 결과 예시</div>
                <div class="verdict-title">🤖 AI 패널 P 승리</div>
                <div class="note-line"><b>오늘의 실제 시장 선택</b><br>SPY -2.8% · TLT +0.9% · SHV +0.3% · GLD +1.4% · DBC +0.5%</div>
                <div class="note-line"><b>플레이어 예측</b>: SPY 30% / TLT 20% / SHV 20% / GLD 20% / DBC 10%</div>
                <div class="note-line"><b>AI 패널 P</b>: SPY 25% / TLT 25% / SHV 25% / GLD 15% / DBC 10%</div>
                <div class="note-line">회차 수익률: 플레이어 <b>-0.35%</b> / AI 패널 P <b>+0.05%</b> → AI 승리 → 상금 -500,000원</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ending_tab:
        st.markdown("#### 최종 엔딩 평가")
        endings = [
            ("🏆", "AI를 이긴 인간 패널", "AI보다 누적 수익률 높음"),
            ("🔍", "시그널 해석 장인", "국면 적합도 최고"),
            ("🚨", "대혼란 회차 생존 패널", "Crisis 방어 성공"),
            ("💘", "직진 예측 성공 패널", "Expansion 수익 극대화"),
            ("🛡", "과몰입 방지 패널", "Euphoria에서 방어 성공"),
            ("🌅", "새로운 시그널 포착 패널", "Bottoming에서 공격 전환 성공"),
            ("🔭", "신중한 관찰형 패널", "손실은 적지만 수익 낮음"),
            ("😵", "과몰입 패널", "공격 과다로 손실 큼"),
        ]
        for emoji, title, cond in endings:
            st.markdown(
                f'<div class="note-line"><b>{emoji} {title}</b> — {cond}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# FINAL ENDING
# ══════════════════════════════════════════════════════════════════════════════
def render_final_ending(history):
    if len(history) < 6:
        return

    player_returns = np.array([float(i["user_return"]) for i in history])
    ai_returns = np.array([float(i["ai_return"]) for i in history])
    player_curve = pd.Series(np.cumprod(1 + player_returns))
    ai_curve = pd.Series(np.cumprod(1 + ai_returns))
    player_cum = float(player_curve.iloc[-1] - 1)
    ai_cum = float(ai_curve.iloc[-1] - 1)
    mdd = float((player_curve / player_curve.cummax() - 1).min())
    avg_signal = float(np.mean([float(i["signal_score"]) for i in history]))
    vol = float(player_returns.std() * np.sqrt(12)) if len(player_returns) > 1 else 0.0
    prize_total = sum(int(i.get("prize_delta", 0)) for i in history)
    gem_total = sum(int(i.get("gem_delta", 0)) for i in history)
    title = ending_title(history)
    pos_cls = "pos" if prize_total >= 0 else "neg"

    st.markdown("## 🏆 최종 패널 평가")
    st.markdown(
        f"""
        <div class="ending-hero">
            <div class="ending-title">{title}</div>
            <div class="ending-sub">
                당신은 6번의 회차 동안 시장의 시그널을 해석하고 ETF 출연진의 다음 선택을 예측했습니다.<br>
                AI 패널 P보다 높은 성과를 냈는지, 국면별 위험관리를 잘했는지, 상금판을 지켜냈는지를 평가합니다.
            </div>
        </div>
        <div class="prize-grid">
            <div class="prize-card">
                <div class="prize-label">Final Prize Board</div>
                <div class="prize-amount {pos_cls}">{format_krw(prize_total)}</div>
                <div class="prize-sub">AI를 이긴 회차 +50만원, 진 회차 -50만원 가상 상금</div>
            </div>
            <div class="prize-card">
                <div class="prize-label">Signal Gems</div>
                <div class="prize-amount {pos_cls}">{gem_total:+d}개</div>
                <div class="prize-sub">누적 승패 시그널 원석 토큰</div>
            </div>
            <div class="prize-card">
                <div class="prize-label">Final Condition</div>
                <div class="prize-amount {pos_cls}">{'상금 방어 ✓' if prize_total >= 0 else '상금 방어 ✗'}</div>
                <div class="prize-sub">6회차 최종 상금판 결과</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("최종 패널 적중률", format_pct(player_cum), delta=format_pct(player_cum - ai_cum))
    c2.metric("시그널 해석력", f"{avg_signal:.0f}점")
    c3.metric("오답 충격도", plain_pct(mdd))
    c4.metric("AI 대비 촉 점수", format_pct(player_cum - ai_cum))
    c5.metric("예측 안정성", plain_pct(vol))

    st.dataframe(
        pd.DataFrame(history).rename(columns={
            "round": "회차", "date": "관찰 날짜", "episode": "실제 회차명",
            "user_return": "플레이어 수익률", "ai_return": "AI 패널 P 수익률",
            "signal_score": "시그널 해석력", "winner": "결과",
            "panel_type": "플레이어 유형", "gem_delta": "시그널 원석", "prize_delta": "상금 변동",
        }),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
df = load_game_data()

for key, default in [
    ("entered", False), ("submitted", False), ("user_weights", None),
    ("user_regime_guess", 1), ("episode_history", []), ("recorded_result_key", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "scenario_date" not in st.session_state:
    st.session_state.scenario_date = pick_random_date(df)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.entered:
    render_entry_screen()
    render_game_guide()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 💘 마켓시그널")
    st.caption("시장의 마음은 오늘 어떤 ETF 출연자에게 향할까?")
    st.progress(min(len(st.session_state.episode_history), 6) / 6)
    st.caption(f"누적 회차 {min(len(st.session_state.episode_history), 6)} / 6")

    current_prize = sum(int(i.get("prize_delta", 0)) for i in st.session_state.episode_history)
    current_gems = sum(int(i.get("gem_delta", 0)) for i in st.session_state.episode_history)
    st.metric("상금판", format_krw(current_prize), delta=f"시그널 원석 {current_gems:+d}개")

    dates = valid_scenario_dates(df)
    date_options = dates[::LOOKBACK_DAYS]
    selected_date = st.selectbox(
        "관찰 날짜",
        options=date_options,
        index=int(np.argmin(np.abs(date_options - st.session_state.scenario_date))),
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
    )
    if pd.Timestamp(selected_date) != st.session_state.scenario_date:
        st.session_state.scenario_date = pd.Timestamp(selected_date)
        st.session_state.submitted = False
        st.session_state.user_weights = None
        st.session_state.recorded_result_key = None

    st.markdown("---")
    st.markdown("#### 💘 ETF 출연진")
    for asset in ASSETS:
        meta = ASSET_META[asset]
        st.markdown(
            f'`{asset}` {meta["emoji"]} **{meta["name"]}**  \n'
            f'<span style="font-size:11px;color:rgba(58,26,40,.50)">{meta["role"]}</span>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN GAME
# ══════════════════════════════════════════════════════════════════════════════
assets = ASSETS
date = st.session_state.scenario_date
row = df.loc[date]
month_window = one_month_window(df, date)
actual_level = infer_signal_regime(row)
actual_regime = REGIMES[actual_level]
ai_w = ai_panel_weights(row, assets, actual_level)
studio_src = studio_image_src()

# ── 히어로 배너 ──
hero_ep = actual_regime["episode"] if st.session_state.submitted else "오늘의 회차 공개"
hero_text = (
    actual_regime["intro"]
    if st.session_state.submitted
    else "시장의 마음은 오늘 어떤 ETF에게 향할까?<br>당신은 인간 패널, 상대는 PPO 기반 AI 패널 P.<br>AI를 이기면 시그널 원석과 50만원을 얻고, 지면 상금판에서 50만원을 잃습니다."
)
risk_val = actual_regime["risk"] if st.session_state.submitted else 48
mood = actual_regime["mood"] if st.session_state.submitted else "🔍 시그널 관찰 중"

st.markdown(
    f"""
    <div class="sys-bar">
        <div class="sys-chip">💘 Market Signal · Human Panel</div>
        <div class="sys-chip">📅 {date.strftime("%Y-%m-%d")}</div>
        <div class="sys-chip">⏱ Signal Window · 1M</div>
        <div class="sys-chip">💎 {format_krw(SIGNAL_GEM_VALUE)} / Round</div>
    </div>
    <div class="hero-banner">
        <div class="hero-bg" style="background-image:url('{REGIME_PHOTOS.get(actual_level, studio_src) if st.session_state.submitted else studio_src}')"></div>
        <div class="hero-overlay"></div>
        <div class="hero-grain"></div>
        <div class="hero-inner">
            <div>
                <div class="hero-badge">{'📺 ' + hero_ep if st.session_state.submitted else '🔍 관찰 중'}</div>
                <div class="hero-title">마켓시그널: AI 패널을 이겨라</div>
                <div class="hero-sub">
                    {mood} &nbsp;·&nbsp; {hero_text}
                </div>
                <div class="hero-risk-wrap">
                    <div class="hero-risk-label">{'실제 국면 긴장도 ' + str(risk_val) + '%' if st.session_state.submitted else '국면 긴장도는 제출 후 공개'}</div>
                    <div class="hero-risk-track">
                        <div class="hero-risk-fill" style="width:{risk_val}%"></div>
                    </div>
                </div>
            </div>
            <div class="hero-photo-wrap">
                <img src="{studio_src}" alt="Signal House">
                <div class="hero-photo-tag">SIGNAL HOUSE</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 구분 레이블 ──
st.markdown(
    '<div style="font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;'
    'color:rgba(58,26,40,.40);margin:0 0 6px">📊 시장 시그널 — 1개월 현황</div>',
    unsafe_allow_html=True,
)
render_signal_cards(row, month_window)

st.markdown(
    '<div style="font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;'
    'color:rgba(58,26,40,.40);margin:8px 0 6px">💘 ETF 출연진 — 시그널 하우스</div>',
    unsafe_allow_html=True,
)
render_cast_cards(assets)

# ── 메인 2컬럼 ──
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("#### 📈 1개월 시그널 흐름")
    st.plotly_chart(make_indicator_chart(month_window), use_container_width=True)

    # 예측 폼
    st.markdown(
        '<div class="predict-card">'
        '<div class="predict-header">'
        '<div class="predict-badge">PANEL PREDICT</div>'
        '<span style="font-size:16px;font-weight:800;color:var(--ink)">시장 마음 예측표</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("market_signal_form"):
        guess = st.selectbox(
            "예상 국면",
            options=list(REGIMES.keys()),
            index=int(st.session_state.user_regime_guess) - 1,
            format_func=regime_option,
        )

        if st.session_state.user_weights is not None:
            defaults = {a: int(round(st.session_state.user_weights.reindex(assets).fillna(0)[a] * 100)) for a in assets}
        else:
            defaults = {a: int(round(recommended_weights(1, assets)[a] * 100)) for a in assets}

        slider_cols = st.columns(len(assets))
        raw_weights = {}
        for col, asset in zip(slider_cols, assets):
            with col:
                meta = ASSET_META[asset]
                st.markdown(
                    f'<div style="font-size:10px;font-weight:800;color:{meta["color"]};'
                    f'text-align:center;margin-bottom:2px;letter-spacing:.05em">{asset}</div>',
                    unsafe_allow_html=True,
                )
                raw_weights[asset] = st.slider(
                    asset,
                    min_value=0, max_value=100,
                    value=defaults[asset], step=5,
                    help=f"{meta['name']} · {meta['tagline']}",
                    label_visibility="collapsed",
                )
                st.markdown(
                    f'<div style="font-size:9px;text-align:center;color:rgba(58,26,40,.65);line-height:1.3;margin-top:2px;word-break:keep-all">{meta["emoji"]} {meta["name"]}</div>',
                    unsafe_allow_html=True,
                )

        total = sum(raw_weights.values())
        normalized_preview = normalize(raw_weights, assets)
        col_prog, col_cap = st.columns([3, 1])
        with col_prog:
            st.progress(min(total, 100) / 100)
        with col_cap:
            st.markdown(
                f'<div style="font-size:12px;font-weight:800;color:{"var(--teal)" if total == 100 else "var(--pink)"};padding-top:4px">{total}%</div>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("💘 시장 마음 예측 제출", use_container_width=True)
        if submitted:
            st.session_state.user_weights = normalized_preview
            st.session_state.user_regime_guess = int(guess)
            st.session_state.submitted = True
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    # 패널 관찰 노트
    notes_html = ['<div class="note-panel">',
                  '<div class="note-kicker">Panel Observation Note</div>']
    if st.session_state.submitted:
        notes_html.append(f'<div class="note-title">실제 국면: {actual_regime["episode"]}</div>')
    else:
        notes_html.append('<div class="note-title">패널 관찰 노트</div>')
    for n in observation_notes(row):
        notes_html.append(f'<div class="note-line">{n}</div>')
    notes_html.append("</div>")
    st.markdown("".join(notes_html), unsafe_allow_html=True)

    st.write("")
    mc = st.columns(2)
    mc[0].metric("US10Y", f"{row['US10Y']:.2f}%")
    mc[1].metric("DXY", f"{row['DXY']:.1f}")
    mc[0].metric("시장 폭", f"{row['Market_Breadth']:.3f}")
    mc[1].metric("Equity/Bond", f"{row['Equity_vs_Bond']:.3f}")

    st.write("")
    st.markdown("#### 🤖 AI 패널 P의 선택")
    if st.session_state.submitted and st.session_state.user_weights is not None:
        st.plotly_chart(make_weight_chart(st.session_state.user_weights, ai_w, assets), use_container_width=True)
    else:
        st.markdown(
            """
            <div class="locked-panel">
                <div class="locked-icon">🔒</div>
                <div class="locked-title">분석 대기실</div>
                <div class="locked-sub">AI 패널 P의 포트폴리오는 예측 제출 후 공개됩니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.submitted and st.session_state.user_weights is not None:
    user_w = st.session_state.user_weights.reindex(assets).fillna(0)
    guess = int(st.session_state.user_regime_guess)
    target_w = recommended_weights(actual_level, assets)

    user_path = portfolio_path(df, date, HORIZON_DAYS, user_w, assets)
    ai_path = portfolio_path(df, date, HORIZON_DAYS, ai_w, assets)
    user_m = metrics(user_path)
    ai_m = metrics(ai_path)
    user_signal = allocation_fit_score(user_w, target_w) * 0.55 + regime_guess_score(actual_level, guess) * 0.45
    ai_signal = allocation_fit_score(ai_w, target_w) * 0.75 + 25
    user_score = round_score(user_m, user_signal)
    ai_score = round_score(ai_m, ai_signal)
    asset_returns = asset_forward_returns(df, date, HORIZON_DAYS, assets)
    round_won = user_score >= ai_score
    winner = "🩷 플레이어 패널 승리" if round_won else "🤖 AI 패널 P 승리"
    gem_delta = 1 if round_won else -1
    prize_delta = SIGNAL_GEM_VALUE if round_won else -SIGNAL_GEM_VALUE
    result_key = f"{date.strftime('%Y-%m-%d')}|{guess}|{'/'.join(f'{user_w[a]:.4f}' for a in assets)}"

    if st.session_state.recorded_result_key != result_key and len(st.session_state.episode_history) < 6:
        st.session_state.episode_history.append({
            "round": len(st.session_state.episode_history) + 1,
            "date": date.strftime("%Y-%m-%d"),
            "level": actual_level,
            "episode": actual_regime["episode"],
            "user_return": user_m["total_return"],
            "ai_return": ai_m["total_return"],
            "signal_score": user_signal,
            "mdd": user_m["mdd"],
            "winner": winner,
            "panel_type": panel_type(user_w),
            "gem_delta": gem_delta,
            "prize_delta": prize_delta,
        })
        st.session_state.recorded_result_key = result_key

    st.markdown("---")
    st.markdown("## 📺 다음 회차 공개")

    prize_board = sum(int(i.get("prize_delta", 0)) for i in st.session_state.episode_history)
    gem_board = sum(int(i.get("gem_delta", 0)) for i in st.session_state.episode_history)

    rc = st.columns(5)
    rc[0].metric("🩷 플레이어 수익률", format_pct(user_m["total_return"]),
                 delta=format_pct(user_m["total_return"] - ai_m["total_return"]))
    rc[1].metric("🤖 AI 패널 P 수익률", format_pct(ai_m["total_return"]))
    rc[2].metric("📡 시그널 해석력", f"{user_signal:.0f}점", delta=f"{user_signal - ai_score:+.0f}")
    rc[3].metric("🏆 회차 점수", f"{user_score:.0f}", delta=f"{user_score - ai_score:+.0f}")
    rc[4].metric("💎 이번 회차 상금", format_krw(prize_delta), delta=f"누적 {format_krw(prize_board)}")

    reveal_l, reveal_r = st.columns([1.05, 0.95], gap="large")

    with reveal_l:
        win_cls = "verdict-win" if round_won else "verdict-lose"
        st.markdown(
            f"""
            <div class="verdict-card {win_cls}">
                <div class="verdict-kicker">실제 시장 국면</div>
                <div class="verdict-title">{actual_regime['episode']} · {actual_regime['finance']}</div>
                <div class="note-line"><b>시장 시그널</b><br>{actual_regime['signal']}</div>
                <div class="note-line"><b>패널 해석</b><br>{actual_regime['panel']}</div>
                <div class="note-line"><b>플레이어 가이드</b><br>{actual_regime['guide']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 다음 회차 자산 곡선")
        st.plotly_chart(make_result_chart(user_path, ai_path), use_container_width=True)

    with reveal_r:
        st.markdown(
            f"""
            <div class="verdict-card {win_cls}">
                <div class="verdict-kicker">패널 판정</div>
                <div class="verdict-title">{winner}</div>
                <div class="note-line"><b>상금판 정산</b><br>{'시그널 원석 1개 획득 💎' if round_won else '시그널 원석 1개 차감'} · 이번 {format_krw(prize_delta)} · 누적 {format_krw(prize_board)} ({gem_board:+d}개)</div>
                <div class="note-line">{result_review(actual_level, guess, user_w, ai_w, user_m['total_return'], ai_m['total_return'])}</div>
                <div class="note-line"><b>플레이어 유형</b> {panel_type(user_w)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            pd.DataFrame({
                "다음 회차 수익률": (asset_returns * 100).round(2).astype(str) + "%",
                "플레이어 예측": (user_w * 100).round(1).astype(str) + "%",
                "AI 패널 P": (ai_w * 100).round(1).astype(str) + "%",
                "국면 추천": (target_w * 100).round(1).astype(str) + "%",
                "출연자": [ASSET_META[a]["name"] for a in assets],
            }, index=assets),
            use_container_width=True,
        )

    # 회차 진행 버튼
    st.markdown("### 회차 진행")
    prog_c = st.columns(3)
    prog_c[0].metric("누적 회차", f"{min(len(st.session_state.episode_history), 6)} / 6")
    prog_c[1].metric("누적 상금판", format_krw(prize_board), delta=f"원석 {gem_board:+d}개")
    prog_c[2].metric("엔딩 조건", "최종 공개" if len(st.session_state.episode_history) >= 6 else "진행 중")

    if len(st.session_state.episode_history) >= 6:
        render_final_ending(st.session_state.episode_history)
        if st.button("🔄 처음부터 다시 입장하기", use_container_width=True):
            for k in ["episode_history", "recorded_result_key", "submitted", "user_weights", "user_regime_guess"]:
                st.session_state[k] = [] if k == "episode_history" else (1 if k == "user_regime_guess" else None if k != "submitted" else False)
            st.session_state.scenario_date = pick_random_date(df)
            st.rerun()
    else:
        if st.button("▶ 다음 회차 이동", use_container_width=True):
            st.session_state.scenario_date = next_episode_date(date, df)
            st.session_state.submitted = False
            st.session_state.user_weights = None
            st.session_state.user_regime_guess = 1
            st.session_state.recorded_result_key = None
            st.rerun()