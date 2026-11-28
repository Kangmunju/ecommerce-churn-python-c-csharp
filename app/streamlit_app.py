"""이커머스 고객 이탈 예측 대시보드 (Python · C · C#).

실행: streamlit run app/streamlit_app.py   (프로젝트 맨 위 폴더에서)

이 앱은 모델을 다시 학습하지 않고, 저장된 CSV 와 소스 코드를 읽어서 보여 준다.
- 기대이익 계산기는 파이썬이 저장한 시험 고객 1,484명의 정답과 예측 확률(app/data/test_pred.csv)을 쓴다.
- 피처와 계수 해석은 reports/logit_coefficients.csv 와 app/data/feature_corr.csv 를 쓴다.
- C 와 C# 은 이 앱 안에서 실행되지 않는다. 저장소에서 실행해 둔 결과(app/data/comparison.csv)와 코드를 보여 준다.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ============================================================
# 경로와 상수
# ============================================================
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent  # 프로젝트 맨 위 폴더 (reports/, src/, c/, csharp/ 가 있는 곳)
REPORTS = ROOT / "reports"
DATA = APP_DIR / "data"
REPO_URL = "https://github.com/Kangmunju/ecommerce-churn-python-c-csharp"

# 쿠폰 발송을 '실행' 으로 권하는 최소 기대이익(파운드), 파이프라인(C#)의 기준과 같음
MIN_WORTH = 100.0

# 원본 거래 수 (reports/DATA_ACCESS.md), 시트가 겹쳐 중복으로 막힌 행 수는 원본과 정제 시작 행 수의 차이로 구함
RAW_ROWS = 1_067_371

# 관측창과 예측창 (src/labels.py 의 CUTOFF, LABEL_END 와 DEVIATIONS.md 의 556일)
OBSERVATION_DAYS = 556
PREDICTION_DAYS = 183

# 색: 정상은 청록, 부풀려진 값은 주황, 일부만 부풀려진 값은 호박색, 중립은 회색
TEAL = "#0F766E"
ORANGE = "#EA580C"
AMBER = "#F59E0B"
GRAY = "#94A3B8"
GREEN = "#15803D"
RED = "#DC2626"
# 언어마다 정한 색 (첫 화면의 이름표와 세 언어 탭에서 같은 색을 씀)
LANG_COLOR = {"Python": "#0F766E", "C": "#BE185D", "C#": "#6D28D9"}

# 피처 20개: (묶음, 이름, 의미)  ·  src/features.py 의 정의를 그대로 옮김
FEATURE_INFO = [
    ("RFM", "recency", "기준일에서 마지막 주문일을 뺀 일수"),
    ("RFM", "frequency", "주문 횟수 (송장 기준)"),
    ("RFM", "monetary", "총 구매액"),
    ("RFM", "avg_order_value", "주문 금액의 평균"),
    ("RFM", "std_order_value", "주문 금액의 표준편차"),
    ("RFM", "max_order_value", "주문 금액의 최댓값"),
    ("RFM", "total_items", "총 구매 수량"),
    ("RFM", "tenure", "기준일에서 첫 주문일을 뺀 일수 (고객이 된 지 얼마나 됐는가)"),
    ("RFM", "active_span", "첫 주문부터 마지막 주문까지의 기간 (일)"),
    ("주문 리듬", "gap_mean", "주문 간격의 평균 (일) · 1회 구매 고객은 tenure 로 채움"),
    ("주문 리듬", "gap_std", "주문 간격의 표준편차 · 1회 구매 고객은 0"),
    ("주문 리듬", "gap_max", "주문 간격의 최댓값 · 1회 구매 고객은 tenure 로 채움"),
    ("주문 리듬", "recency_ratio", "recency 를 gap_mean 으로 나눈 값 · 그 고객의 평소 리듬보다 얼마나 늦었는가 (상한 50)"),
    ("최근 90일", "orders_90d", "기준일 직전 90일의 주문 수"),
    ("최근 90일", "revenue_90d", "기준일 직전 90일의 구매액"),
    ("최근 90일", "share_90d", "최근 90일 구매액을 총 구매액으로 나눈 값"),
    ("탐색·반품", "n_products", "구매한 상품 종류 수"),
    ("탐색·반품", "n_returns", "관측창 안의 반품(취소 전표) 송장 수"),
    ("탐색·반품", "return_rate", "반품 수를 주문 수로 나눈 값 (상한 1)"),
    ("국가", "is_uk", "마지막 주문의 국가가 영국이면 1"),
]

st.set_page_config(page_title="이커머스 고객 이탈 예측 | Python · C · C#", layout="wide")

# 글자 크기, 간격, 카드 모양을 맞춤
# - 본문 폭을 1100px 로 제한해 한 줄이 너무 길어지지 않게 함
# - 카드 테두리와 배경은 회색 투명도로 지정해 밝은 화면과 어두운 화면에서 모두 보이게 함
st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 4rem; }
    h1 { font-size: 2.1rem !important; }
    h3 { font-size: 1.4rem !important; padding-top: 0.5rem; }
    h4 { font-size: 1.12rem !important; padding-top: 1.2rem; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.75; }
    [data-testid="stCaptionContainer"] { font-size: 0.92rem; line-height: 1.65; opacity: 0.8; }

    .hero-q { font-size: 1.45rem; font-weight: 600; margin: 0.2rem 0 0.6rem 0; }
    .hero-sub { font-size: 1.02rem; line-height: 2; }
    .chip { display: inline-block; padding: 1px 10px; border-radius: 999px; font-weight: 600;
            font-size: 0.92rem; color: #fff; }
    .arrow { opacity: 0.55; margin: 0 4px; }

    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
                 gap: 12px; margin: 0.4rem 0 1.2rem 0; }
    .stat { border: 1px solid rgba(128,128,128,0.28); border-radius: 10px; padding: 14px 16px;
            background: rgba(128,128,128,0.06); }
    .stat-label { font-size: 0.9rem; opacity: 0.75; }
    .stat-value { font-size: 1.9rem; font-weight: 600; line-height: 1.35; }
    .stat-sub { font-size: 0.85rem; opacity: 0.7; min-height: 1.3rem; }
    .stat ul { margin: 0.3rem 0 0 0; padding-left: 1.2rem; line-height: 1.7; }

    .timeline { display: flex; width: 100%; border-radius: 10px; overflow: hidden; margin: 0.6rem 0 0.4rem 0; }
    .timeline .seg { padding: 14px 16px; color: #fff; line-height: 1.5; }
    .timeline .seg b { font-size: 1.05rem; }
    .timeline .seg span { font-size: 0.88rem; opacity: 0.92; }

    table.grid { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
    table.grid th { text-align: left; font-weight: 600; padding: 10px 12px;
                    border-bottom: 2px solid rgba(128,128,128,0.35); }
    table.grid td { text-align: left; vertical-align: top; padding: 10px 12px;
                    border-bottom: 1px solid rgba(128,128,128,0.2); line-height: 1.6; }
    .tag { display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: 0.88rem;
           border-left: 4px solid #888; background: rgba(128,128,128,0.10); }
    .tag.none { opacity: 0.55; border-left-style: dashed; }
    .warn { color: #EA580C; font-weight: 600; }
    table.grid .nw { white-space: nowrap; }

    [role="tablist"] { gap: 0; }
    [data-testid="stTab"] { flex: 1 1 0; justify-content: center; padding: 1rem 0.5rem; height: auto; }
    [data-testid="stTab"] * { font-size: 1.12rem !important; font-weight: 600; }
    @media (max-width: 640px) {
        [data-testid="stTab"] { flex: 0 0 auto; padding: 0.7rem 0.9rem; }
        [data-testid="stTab"] * { font-size: 1rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 데이터 읽기 (한 번 읽은 결과는 캐시에 둠)
# ============================================================
@st.cache_data
def load_csv(path: str, as_text: bool = False) -> pd.DataFrame:
    """CSV 를 읽음. as_text 가 True 면 모든 칸을 글자 그대로 읽어 0.30 이 0.3 으로 바뀌지 않게 함."""
    if as_text:
        return pd.read_csv(path, dtype=str)
    return pd.read_csv(path)


@st.cache_data
def load_assumptions() -> dict:
    with open(REPORTS / "a5_assumptions.json", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Path) -> list[str]:
    """파일이 없으면 빈 목록을 돌려줌 (배포 환경에서 일부 파일이 없어도 앱이 죽지 않게 함)."""
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def extract(path: Path, start: str, end: str) -> str | None:
    """start 가 들어 있는 줄부터 end 가 들어 있는 줄 바로 앞까지를 잘라 옴.

    줄 번호가 아니라 글자를 기준으로 찾기 때문에 코드가 조금 바뀌어도 대부분 맞게 동작함.
    """
    lines = read_lines(path)
    first = next((i for i, line in enumerate(lines) if start in line), None)
    if first is None:
        return None
    last = next((i for i in range(first + 1, len(lines)) if end in lines[i]), None)
    if last is None:
        return None
    return textwrap.dedent("\n".join(lines[first:last])).rstrip()


def money(value: float) -> str:
    """파운드 금액을 글자로 바꿈. 손해는 -£130 처럼 마이너스를 앞에 붙임."""
    sign = ""
    if value < 0:
        sign = "-"
    return f"{sign}£{abs(value):,.0f}"


def signed(value: float, digits: int = 2) -> str:
    """부호를 항상 붙여 +0.44, -1.47 처럼 적음."""
    return f"{value:+.{digits}f}"


def html(text: str) -> None:
    """HTML 조각을 그림. 줄바꿈과 빈 줄이 있으면 마크다운이 코드로 읽기 때문에 한 줄로 이어 붙임."""
    one_line = "".join(line.strip() for line in text.splitlines())
    st.markdown(one_line, unsafe_allow_html=True)


def lang_chip(name: str) -> str:
    return f'<span class="chip" style="background:{LANG_COLOR[name]}">{name}</span>'


def stat_card(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    """이름, 큰 숫자, 보조 설명을 한 카드에 묶음. 이름과 값이 같은 카드 안에 있어 서로 이어져 보임."""
    style = ""
    if color is not None:
        style = f' style="color:{color}"'
    return (
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value"{style}>{value}</div>'
        f'<div class="stat-sub">{sub}</div></div>'
    )


def stat_grid(cards: list[str]) -> None:
    """카드들을 한 줄 격자로 그림. 격자가 카드 높이를 같게 맞춰 줘서 글이 길어도 높이가 들쭉날쭉하지 않음."""
    html('<div class="stat-grid">' + "".join(cards) + "</div>")


def styled(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    """차트의 높이(픽셀)와 글자 크기를 맞춤."""
    return (
        chart.properties(height=height)
        .configure_axis(labelFontSize=13, titleFontSize=13)
        .configure_legend(labelFontSize=13, titleFontSize=13)
    )


# ============================================================
# 기대이익 계산
# ============================================================
def profit_curve(y: pd.Series, p: pd.Series, value_tp: float, value_fp: float) -> pd.DataFrame:
    """임계값 0.05 ~ 0.95(0.05 간격)마다 쿠폰을 보냈을 때의 기대이익을 구함.

    예측 확률이 임계값 이상인 고객에게 보냄.
    맞게 보낸 1건(TP)은 value_tp, 헛되이 보낸 1건(FP)은 value_fp 만큼 이익이 됨.
    """
    rows = []
    for k in range(1, 20):
        threshold = k * 5 / 100
        sent = p >= threshold
        tp = int((sent & (y == 1)).sum())
        fp = int((sent & (y == 0)).sum())
        rows.append(
            {
                "임계값": threshold,
                "발송": tp + fp,
                "TP": tp,
                "FP": fp,
                "기대이익": tp * value_tp + fp * value_fp,
            }
        )
    return pd.DataFrame(rows)


def advice_of(best_profit: float) -> str:
    """이익이 0 이하면 하지 말 것, 기준 이상이면 실행, 그 사이는 사실상 손익분기."""
    if best_profit <= 0.0:
        return "하지 말 것"
    if best_profit >= MIN_WORTH:
        return "실행"
    return "사실상 손익분기"


# ============================================================
# 화면 맨 위
# ============================================================
st.title("이커머스 고객 이탈 예측")
html(
    f"""
    <div class="hero-q">이 고객은 앞으로 6개월 안에 다시 살까?</div>
    <div class="hero-sub">
      {lang_chip("Python")}<span class="arrow">→</span>{lang_chip("C")}<span class="arrow">→</span>{lang_chip("C#")}
      &nbsp;세 언어로 같은 문제를 풀고, 결과가 서로 맞는지 비교
    </div>
    """
)
st.caption(
    f"코드와 보고서: [GitHub 저장소]({REPO_URL}) · "
    "이 앱은 저장된 결과를 읽어 보여 주며 모델은 다시 학습하지 않음"
)

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["프로젝트 개요", "기대이익 계산기", "함정 4가지", "세 언어로 구현", "방법과 한계"]
)

# ============================================================
# 탭 0. 프로젝트 개요
# ============================================================
with tab0:
    st.subheader("프로젝트 개요")
    st.markdown(
        "**데이터** · 영국 온라인 선물용품 소매업체의 실제 거래 106만 행 (2009-12 → 2011-12)  \n"
        "**문제** · 고객의 과거 구매 기록만 보고, 앞으로 6개월 안에 다시 살지 맞히기  \n"
        "**정답** · 그 6개월 동안 구매가 한 번도 없으면 '이탈'"
    )

    # ---- 1. 문제 정의 ----
    st.markdown("#### 1. 문제 정의")
    html(
        f"""
        <div class="timeline">
          <div class="seg" style="flex:{OBSERVATION_DAYS}; background:{TEAL}">
            <b>관측창 {OBSERVATION_DAYS}일</b><br><span>2009-12-01 → 2011-06-09 · 피처를 만드는 구간</span>
          </div>
          <div class="seg" style="flex:{PREDICTION_DAYS}; background:{ORANGE}">
            <b>예측창 {PREDICTION_DAYS}일</b><br><span>2011-06-10 → 2011-12-09 · 정답을 정하는 구간</span>
          </div>
        </div>
        """
    )
    st.caption("기준일 2011-06-10 · 예측창에 구매가 한 번도 없으면 이탈 · 기준일 이후 데이터는 피처에 한 줄도 쓰지 않음")

    a4_rows = load_csv(str(REPORTS / "a4_survivorship.csv"))
    customers = int(a4_rows.iloc[0]["고객 수"])
    churn_rate = float(a4_rows.iloc[0]["이탈률"])
    churned = round(customers * churn_rate)
    test_count = len(load_csv(str(DATA / "test_pred.csv")))
    stat_grid(
        [
            stat_card("분석 대상 고객", f"{customers:,}명", "관측창에 1건 이상 구매"),
            stat_card("이탈 고객", f"{churned:,}명", f"이탈률 {churn_rate * 100:.1f}%"),
            stat_card("학습 / 시험", f"{customers - test_count:,} / {test_count:,}", "무작위 분할 (7 : 3)"),
        ]
    )

    # ---- 2. 데이터 흐름 ----
    st.markdown("#### 2. 데이터 흐름: 정제 퍼널")
    clean_log = load_csv(str(REPORTS / "clean_log.csv"))
    dedup_rows = int(clean_log.iloc[0]["before"])  # 시트가 겹쳐 중복을 막은 뒤의 행 수
    funnel = [("원본", RAW_ROWS, 0), ("중복 제거", dedup_rows, RAW_ROWS - dedup_rows)]
    for _, row in clean_log.iterrows():
        step_name = re.sub(r"^\d+\.\s*", "", str(row["step"]))
        funnel.append((step_name, int(row["after"]), int(row["dropped"])))
    funnel_df = pd.DataFrame(funnel, columns=["단계", "행 수", "줄어든 행"])
    biggest = funnel_df["줄어든 행"].idxmax()  # 가장 많이 줄어든 단계만 주황으로 강조
    funnel_df = funnel_df.assign(강조=["큰 손실" if i == biggest else "일반" for i in range(len(funnel_df))])
    funnel_df = funnel_df.assign(라벨=funnel_df["행 수"].map(lambda v: f"{v:,}"))

    base = alt.Chart(funnel_df).encode(y=alt.Y("단계:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(clip=True).encode(
        x=alt.X("행 수:Q", title="거래 행 수", scale=alt.Scale(domain=[0, 1_350_000]), axis=alt.Axis(format="~s")),
        color=alt.Color("강조:N", scale=alt.Scale(domain=["일반", "큰 손실"], range=[TEAL, ORANGE]), legend=None),
        tooltip=["단계", alt.Tooltip("행 수:Q", format=",d"), alt.Tooltip("줄어든 행:Q", format=",d")],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=13).encode(x="행 수:Q", text="라벨:N")
    st.altair_chart(styled(bars + labels, 300), width="stretch")
    lost = int(funnel_df.loc[biggest, "줄어든 행"])
    st.caption(
        f"가장 크게 줄어든 단계 · {funnel_df.loc[biggest, '단계']} −{lost:,}행 (매출 13.1% 포기) · "
        "이탈은 고객 단위 문제라 고객 번호가 없는 거래는 분석에서 제외"
    )

    # ---- 3. 핵심 결과 ----
    st.markdown("#### 3. 핵심 결과")
    scores = load_csv(str(REPORTS / "model_scores.csv"))
    cv = load_csv(str(REPORTS / "cv_scores.csv")).set_index("model")
    logit_auc = float(scores.loc[scores["model"].str.contains("로지스틱"), "AUC"].iloc[0])
    mlp_auc = float(scores.loc[scores["model"].str.contains("MLP"), "AUC"].iloc[0])
    stat_grid(
        [
            stat_card(
                "로지스틱 AUC",
                f"{logit_auc:.4f}",
                f"5-fold {cv.loc['로지스틱', 'mean']:.4f} ± {cv.loc['로지스틱', 'std']:.4f}",
            ),
            stat_card(
                "MLP AUC", f"{mlp_auc:.4f}", f"5-fold {cv.loc['MLP', 'mean']:.4f} ± {cv.loc['MLP', 'std']:.4f}"
            ),
            stat_card("최적 기대이익", "£6,969", "임계값 0.30 · 기본 0.50 은 £6,236"),
            stat_card("이탈 정의만 바꾸면", "AUC 1.0000", "원래 0.7818 · 함정 1", ORANGE),
        ]
    )

    scores = scores.assign(모델=scores["model"].map(lambda v: re.sub(r"^\d+\.\s*", "", str(v)).replace("'", "")))
    scores = scores.assign(구분=["기준선", "기준선", "모델", "모델"])
    scores = scores.assign(라벨=scores["AUC"].map(lambda v: f"{v:.4f}"))
    base = alt.Chart(scores).encode(y=alt.Y("모델:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)))
    bars = base.mark_bar(clip=True).encode(
        x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.5, 0.85])),
        color=alt.Color("구분:N", scale=alt.Scale(domain=["기준선", "모델"], range=[GRAY, TEAL]), legend=None),
        tooltip=["모델", alt.Tooltip("AUC:Q", format=".4f")],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=13).encode(x="AUC:Q", text="라벨:N")
    st.altair_chart(styled(bars + labels, 210), width="stretch")
    st.caption("회색 · 기준선(다수 클래스, 규칙 하나)  /  청록 · 모델  /  MLP 는 로지스틱과 구분되지 않음 (시드 10개 평균 0.7836 vs 0.7818)")
    st.divider()

    # ---- 4. 무엇이 이탈을 만드는가 ----
    st.markdown("#### 4. 무엇이 이탈과 이어지는가: 피처 20개와 로지스틱 계수")
    st.write(
        "피처 20개는 모두 **기준일 이전 거래** 로만 계산 · 로지스틱 회귀의 계수는 "
        "**표준화한 값 기준(1 표준편차 변할 때의 변화)** 이라 피처끼리 크기를 비교할 수 있음"
    )

    coef = pd.read_csv(REPORTS / "logit_coefficients.csv", header=0, names=["feature", "coef"])
    corr = load_csv(str(DATA / "feature_corr.csv")).rename(columns={"corr_with_churn": "corr"})
    info = pd.DataFrame(FEATURE_INFO, columns=["묶음", "feature", "의미"])
    table = coef.merge(corr, on="feature").merge(info, on="feature")
    table = table.assign(
        방향=table["coef"].map(lambda v: "이탈 쪽" if v > 0 else "유지 쪽"),
        뒤집힘=[(c > 0) != (r > 0) and abs(r) >= 0.05 for c, r in zip(table["coef"], table["corr"])],
    )

    chart = (
        alt.Chart(table)
        .mark_bar(clip=True)
        .encode(
            y=alt.Y("feature:N", sort=alt.EncodingSortField("coef", order="descending"), title=None),
            x=alt.X("coef:Q", title="로지스틱 계수 (표준화) · 오른쪽일수록 이탈 확률을 높임"),
            color=alt.Color("방향:N", scale=alt.Scale(domain=["이탈 쪽", "유지 쪽"], range=[ORANGE, TEAL]), legend=None),
            tooltip=[
                alt.Tooltip("feature:N", title="피처"),
                alt.Tooltip("의미:N"),
                alt.Tooltip("coef:Q", title="계수", format="+.3f"),
                alt.Tooltip("corr:Q", title="이탈과의 단독 상관", format="+.2f"),
            ],
        )
    )
    st.altair_chart(styled(chart, 520), width="stretch")
    st.caption("주황 · 이탈 확률을 높이는 쪽  /  청록 · 이탈 확률을 낮추는 쪽 (남아 있는 쪽)")

    top = table.reindex(table["coef"].abs().sort_values(ascending=False).index).head(3)
    insight_cards = []
    for _, row in top.iterrows():
        direction = "이탈 확률이 높아짐" if row["coef"] > 0 else "이탈 확률이 낮아짐"
        insight_cards.append(
            stat_card(
                row["feature"],
                signed(row["coef"]),
                f"{row['의미']} → 값이 클수록 {direction}",
                ORANGE if row["coef"] > 0 else TEAL,
            )
        )
    st.markdown("**계수가 큰 피처 3개**")
    stat_grid(insight_cards)

    flipped = table[table["뒤집힘"]]
    flipped = flipped.reindex(flipped["coef"].abs().sort_values(ascending=False).index)  # 계수가 큰 순서
    flip_items = "".join(
        f"<li><b>{r['feature']}</b> · 단독 상관 {signed(r['corr'])} → 계수 {signed(r['coef'])}</li>"
        for _, r in flipped.head(3).iterrows()
    )
    if len(flipped) > 3:
        flip_items += f"<li>그 밖에 {len(flipped) - 3}개 · 아래 표의 비고 열에서 확인</li>"
    html(
        f"""
        <div class="stat" style="border-left:4px solid {ORANGE}">
          <div class="stat-label"><b>읽을 때 주의</b></div>
          <ul>
            <li><b>상관이지 인과가 아님</b> · 최근 주문이 많은 고객이 남는다는 뜻이지, 주문을 늘리면 이탈이 줄어든다는 뜻이 아님</li>
            <li><b>피처끼리 겹치면 부호가 뒤집힘</b> · 이탈과의 단독 상관과 계수의 부호가 반대인 피처가 {len(flipped)}개 있음 (계수가 큰 3개만 표시)</li>
          </ul>
          <ul style="margin-top:0">{flip_items}</ul>
          <div class="stat-sub">다른 피처와 함께 넣으면 그 피처만의 몫이 달라지므로, 개별 계수를 독립된 영향으로 읽으면 안 됨</div>
        </div>
        """
    )

    with st.expander("피처 20개 전체 보기 (묶음, 의미, 계수, 단독 상관)"):
        full = table.reindex(table["coef"].abs().sort_values(ascending=False).index)
        rows_html = ""
        for _, r in full.iterrows():
            note = ""
            if r["뒤집힘"]:
                note = '<span class="warn">부호 반대</span>'
            rows_html += (
                f'<tr><td class="nw">{r["묶음"]}</td><td class="nw"><b>{r["feature"]}</b></td><td>{r["의미"]}</td>'
                f'<td class="nw">{signed(r["coef"])}</td><td class="nw">{signed(r["corr"])}</td><td class="nw">{note}</td></tr>'
            )
        html(
            f"""
            <table class="grid">
              <tr><th class="nw">묶음</th><th class="nw">피처</th><th>의미</th><th class="nw">계수</th><th class="nw">단독 상관</th><th class="nw">비고</th></tr>
              {rows_html}
            </table>
            """
        )
        st.caption("계수 크기순 · 계수는 표준화 기준 · 단독 상관은 피처 하나와 이탈(0/1)의 상관계수 (피어슨)")

# ============================================================
# 탭 1. 기대이익 계산기
# ============================================================
with tab1:
    st.subheader("임계값은 돈이 정한다")
    st.write(
        "이탈 확률이 임계값 이상인 고객에게만 쿠폰 발송 → 임계값에 따라 기대이익이 달라지는 모습을 직접 확인  \n"
        "쿠폰 반응률·마진·쿠폰 원가는 데이터로 알 수 없는 **가정** → 값을 바꿔 가며 결론이 뒤집히는지 확인"
    )

    assumptions = load_assumptions()
    retained = float(assumptions["R_RETAINED_median_measured"])  # 유지 고객의 예측창 매출 중앙값 (실측)
    test = load_csv(str(DATA / "test_pred.csv"))

    cards_area = st.container()  # 카드는 아래에서 값을 계산한 뒤 이 자리에 그림
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**가정 바꾸기**")
        response = st.slider("쿠폰 반응률", 0.01, 0.40, float(assumptions["RESPONSE"]), 0.01, format="%.2f")
        margin = st.slider("마진", 0.05, 0.80, float(assumptions["MARGIN"]), 0.05, format="%.2f")
        coupon = st.slider("쿠폰 원가 (파운드)", 0.5, 20.0, float(assumptions["COUPON_COST"]), 0.5)
        st.caption(f"실측값 · 유지 고객의 예측창 매출 중앙값 £{retained:,.2f}")

    value_tp = response * retained * margin - coupon  # 맞게 보낸 쿠폰 1장의 이익
    value_fp = -coupon  # 헛되이 보낸 쿠폰 1장의 이익
    curve = profit_curve(test["y"], test["p_logit"], value_tp, value_fp)
    best = curve.loc[curve["기대이익"].idxmax()]  # 이익이 같으면 앞쪽(작은 임계값)
    at_half = curve.loc[curve["임계값"] == 0.50].iloc[0]
    verdict = advice_of(float(best["기대이익"]))

    best_profit = float(best["기대이익"])
    half_profit = float(at_half["기대이익"])
    gap = best_profit - half_profit
    if gap > 0:
        gap_text = f"최적보다 {money(gap)} 낮음"
    else:
        gap_text = "최적과 같음"

    verdict_color = {"실행": GREEN, "사실상 손익분기": AMBER, "하지 말 것": RED}[verdict]
    verdict_reason = {
        "실행": f"이익 £{MIN_WORTH:,.0f} 이상",
        "사실상 손익분기": f"이익 £{MIN_WORTH:,.0f} 미만",
        "하지 말 것": "이익 0 이하",
    }[verdict]

    with cards_area:
        stat_grid(
            [
                stat_card("최적 임계값", f"{best['임계값']:.2f}", "이익이 가장 큰 기준"),
                stat_card("최적의 기대이익", money(best_profit), f"발송 {int(best['발송']):,}명"),
                stat_card("임계값 0.50 일 때", money(half_profit), f"발송 {int(at_half['발송']):,}명 · {gap_text}"),
                stat_card("권고", verdict, verdict_reason, verdict_color),
            ]
        )

    with right:
        line = (
            alt.Chart(curve)
            .mark_line(color=TEAL, point=alt.OverlayMarkDef(color=TEAL))
            .encode(
                x=alt.X("임계값:Q", title="임계값 (이탈 확률이 이 값 이상이면 쿠폰 발송)"),
                y=alt.Y("기대이익:Q", title="기대이익 (파운드)"),
                tooltip=["임계값", "발송", "TP", "FP", alt.Tooltip("기대이익:Q", format=",.0f")],
            )
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color=GRAY).encode(y="y:Q")
        marker = (
            alt.Chart(pd.DataFrame([best]))
            .mark_point(size=260, filled=True, opacity=1, color=ORANGE, stroke="white", strokeWidth=2)
            .encode(x="임계값:Q", y="기대이익:Q")
        )
        st.altair_chart(styled(line + zero + marker, 340), width="stretch")
        st.caption("주황 점 · 이익이 가장 큰 임계값  /  점선 · 이익 0")

    with st.expander("기본 가정에서 파이썬이 확인한 결과"):
        st.markdown(
            "- 시험 고객에서 고른 최적 임계값 **0.30** (£6,969) · 관례적인 0.50 은 £6,236\n"
            "- 학습 고객의 out-of-fold 확률로 임계값을 골라도 **0.25 에서 £6,909** · 이득 유지\n"
            "- 슬라이더를 기본값(반응률 0.10, 마진 0.30, 쿠폰 원가 5.0)에 두면 위 숫자와 동일"
        )
        sens = load_csv(str(REPORTS / "a5_sensitivity.csv"))
        sens_view = pd.DataFrame(
            {
                "쿠폰 반응률": (sens["반응률"] * 100).round(0).astype(int).astype(str) + "%",
                "쿠폰 원가": sens["쿠폰원가"].map(lambda v: f"£{v:,.2f}"),
                "최적 임계값": sens["최적 임계값"].map(lambda v: f"{v:.2f}"),
                "발송": sens["발송"].map(lambda v: f"{v:,}명"),
                "기대이익": sens["기대이익"].map(lambda v: money(float(v))),
                "권고": sens["권고"],
            }
        )
        st.markdown("**가정을 바꿨을 때의 결론 (파이썬 결과)**")
        st.dataframe(sens_view, hide_index=True, width="stretch")

# ============================================================
# 탭 2. 함정 4가지
# ============================================================
with tab2:
    st.subheader("이탈 예측의 함정 4가지")
    st.write("같은 데이터·같은 모델에서 **정의, 시점, 모집단, 전처리** 만 바꿨을 때의 변화")

    st.markdown("#### 1. 이탈 '정의' 를 바꾸면")
    st.write("recency 로 라벨을 만들면 모델이 정답을 베낌 → AUC 1.0 근접 · 관련 피처 8개를 지워도 0.83 으로 부풀려짐")
    a2 = load_csv(str(REPORTS / "a2_definition_type.csv"))
    a2 = a2.assign(구분=a2["정의"].str[0])  # A 는 정상, B 는 정의를 바꾼 것, C 는 일부만 제거한 것
    chart = (
        alt.Chart(a2)
        .mark_bar(clip=True)
        .encode(
            y=alt.Y("정의:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
            x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.5, 1.0])),
            color=alt.Color("구분:N", scale=alt.Scale(domain=["A", "B", "C"], range=[TEAL, ORANGE, AMBER]), legend=None),
            tooltip=["정의", alt.Tooltip("AUC:Q", format=".4f"), alt.Tooltip("이탈률:Q", format=".3f")],
        )
    )
    st.altair_chart(styled(chart, 270), width="stretch")
    st.caption("청록 · 정상 정의  /  주황 · 정의를 바꿔 정답을 베낀 경우  /  호박색 · 관련 피처를 지웠는데도 부풀려진 경우")
    st.divider()

    st.markdown("#### 2. 시점 누수")
    st.write("같은 피처 8개 · 기준일 이전까지만 0.7829 vs 전체 기간 0.9996 → 차이 +0.2167 은 공짜로 생긴 값")
    a3 = load_csv(str(REPORTS / "a3_leakage.csv"))
    a3 = a3.assign(상태=["정상", "누수"])
    chart = (
        alt.Chart(a3)
        .mark_bar(clip=True)
        .encode(
            y=alt.Y("피처를 만든 구간:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
            x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.5, 1.0])),
            color=alt.Color("상태:N", scale=alt.Scale(domain=["정상", "누수"], range=[TEAL, ORANGE]), legend=None),
            tooltip=["피처를 만든 구간", alt.Tooltip("AUC:Q", format=".4f")],
        )
    )
    st.altair_chart(styled(chart, 210), width="stretch")
    st.divider()

    st.markdown("#### 3. 생존편향 (모집단을 바꾸면)")
    st.write("1회 구매 고객 제외 시 단일 분할 AUC 0.7818 → 0.8078 (상승) · 5-fold CV AUC 0.8044 → 0.7835 (하락) → 분할 노이즈를 개선으로 읽으면 안 됨")
    a4 = load_csv(str(REPORTS / "a4_survivorship.csv"))
    a4_view = pd.DataFrame(
        {
            "모집단": a4["모집단"],
            "고객 수": a4["고객 수"].map(lambda v: f"{v:,}명"),
            "이탈률": a4["이탈률"].map(lambda v: f"{v * 100:.1f}%"),
            "단일 분할 AUC": a4["AUC"].map(lambda v: f"{v:.4f}"),
            "5-fold CV AUC": a4["CV AUC"].map(lambda v: f"{v:.4f}"),
            "CV 표준편차": a4["CV 표준편차"].map(lambda v: f"{v:.4f}"),
        }
    )
    st.dataframe(a4_view, hide_index=True, width="stretch")
    st.divider()

    st.markdown("#### 4. MLP 를 표에 올릴 때")
    st.write("표준화를 빼면 MLP −0.0697 · 로지스틱은 거의 그대로  /  시드 10개 평균 MLP 0.7836 ≈ 로지스틱 0.7818 (구분 불가)")
    a6 = load_csv(str(REPORTS / "a6_scaling.csv"))
    chart = (
        alt.Chart(a6)
        .mark_bar(clip=True)
        .encode(
            x=alt.X("모델:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="표준화:N",
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.6, 0.85])),
            color=alt.Color("표준화:N", scale=alt.Scale(domain=["있음", "없음"], range=[TEAL, ORANGE])),
            tooltip=["모델", "표준화", alt.Tooltip("AUC:Q", format=".4f")],
        )
    )
    st.altair_chart(styled(chart, 330), width="stretch")

# ============================================================
# 탭 3. 세 언어로 구현
# ============================================================
with tab3:
    st.subheader("같은 문제를 세 언어로")
    st.write(
        "Python 으로 먼저 만들어 정답지로 삼음 → C 로 라이브러리 없이 직접 구현하며 한계를 기록 → C# 에서 그 한계를 해결"
    )

    # ---- 1. 구현 현황 ----
    st.markdown("#### 1. 단계별 구현 현황")
    stages = [
        (
            "읽기 (정제는 Python 에서만)",
            ["src/clean.py"],
            ["c/src/csv.c", "c/src/datetime.c", "c/src/transactions.c", "c/src/returns.c"],
            ["csharp/src/Churn.Core/Data/CsvLoader.cs"],
        ),
        ("이탈 라벨", ["src/labels.py"], ["c/src/labels.c"], ["csharp/src/Churn.Core/Labels/LabelBuilder.cs"]),
        ("피처 20개", ["src/features.py"], ["c/src/features.c"], ["csharp/src/Churn.Core/Features/FeatureBuilder.cs"]),
        (
            "표준화 + 로지스틱",
            ["src/train.py"],
            ["c/src/logistic.c"],
            [
                "csharp/src/Churn.Core/Modeling/StandardScaler.cs",
                "csharp/src/Churn.Core/Modeling/LogisticRegression.cs",
            ],
        ),
        ("평가 지표", ["src/train.py"], ["c/src/metrics.c"], ["csharp/src/Churn.Core/Evaluation/MetricsCalculator.cs"]),
        (
            "기대이익",
            ["analysis/a5_profit.py"],
            ["c/src/profit.c"],
            ["csharp/src/Churn.Core/Evaluation/ProfitCalculator.cs"],
        ),
        (
            "교차검증 · OOF 임계값 · 보정 · 민감도",
            ["analysis/a5_profit.py"],
            [],
            [
                "csharp/src/Churn.Core/Validation/CrossValidator.cs",
                "csharp/src/Churn.Core/Evaluation/ThresholdSelector.cs",
                "csharp/src/Churn.Core/Evaluation/CalibrationCalculator.cs",
                "csharp/src/Churn.Core/Evaluation/SensitivityAnalyzer.cs",
            ],
        ),
        (
            "MLP",
            ["src/train.py", "analysis/a6_mlp_pitfalls.py"],
            [],
            ["csharp/src/Churn.Core/Modeling/MlpNetwork.cs", "csharp/src/Churn.Core/Modeling/MlpClassifier.cs"],
        ),
    ]

    def stage_cell(language: str, files: list[str]) -> str:
        """구현한 단계는 언어 색 막대가 붙은 줄 수 표시, 구현하지 않은 단계는 흐린 '구현 안 함' 표시."""
        if not files:
            return '<span class="tag none">구현 안 함</span>'
        total = sum(len(read_lines(ROOT / f)) for f in files)
        names = ", ".join(Path(f).name for f in files)
        return f'<span class="tag" style="border-left-color:{LANG_COLOR[language]}" title="{names}">{total:,}줄</span>'

    rows_html = ""
    for name, py_files, c_files, cs_files in stages:
        rows_html += (
            f"<tr><td>{name}</td><td>{stage_cell('Python', py_files)}</td>"
            f"<td>{stage_cell('C', c_files)}</td><td>{stage_cell('C#', cs_files)}</td></tr>"
        )
    html(
        f"""
        <table class="grid">
          <tr><th>단계</th><th>{lang_chip("Python")}</th><th>{lang_chip("C")}</th><th>{lang_chip("C#")}</th></tr>
          {rows_html}
        </table>
        """
    )
    st.caption(
        "줄 수는 주석과 빈 줄 포함 · 파일 이름은 칸 위에 마우스를 올리면 표시 · "
        "Python 은 pandas·scikit-learn 사용, C 는 라이브러리 없이 직접 구현이라 단순 비교는 곤란"
    )

    # ---- 2. 한계에서 해결로 ----
    st.markdown("#### 2. C 에서 부딪힌 한계 → C# 에서의 변화")
    html(
        """
        <table class="grid">
          <tr><th style="width:18%">항목</th><th style="width:41%">C (직접 구현)</th><th style="width:41%">C# (확장)</th></tr>
          <tr><td>날짜 해석</td>
              <td><code>strptime</code> 없음 · <code>mktime</code> 은 시간대 때문에 -32,400초 오차 → 직접 계산</td>
              <td><code>DateTime.ParseExact</code> 로 해결 · 직접 계산 불필요</td></tr>
          <tr><td>고객별 묶기</td>
              <td>해시맵 없음 → 정렬 + 이진탐색 (선형 탐색 0.76초 → 0.098초)</td>
              <td><code>Dictionary</code> 로 한 번 순회</td></tr>
          <tr><td>타입별 코드</td>
              <td>제네릭 없음 → 거래·반품 표를 같은 코드로 복사</td>
              <td>제네릭 <code>ReadRows&lt;T&gt;</code> 하나로 통합</td></tr>
          <tr><td>시험</td>
              <td>시험 틀 없음 → <code>main.c</code> 에 섞어 단계마다 교체</td>
              <td>xUnit 시험 115개를 실행 프로그램과 분리</td></tr>
          <tr><td>읽기 속도·메모리</td>
              <td>0.7-0.8초 · 최대 74.6MB</td>
              <td><span class="warn">1.3-1.5초 · 약 144MB</span> (C# 이 더 느리고 더 씀)</td></tr>
        </table>
        """
    )

    # ---- 3. 같은 일을 세 언어로 ----
    st.markdown("#### 3. 같은 일을 세 언어로: 고객별로 묶어 이탈 라벨 만들기")
    st.write("거래 77만 줄을 고객 번호별로 묶어 첫 구매·마지막 구매·주문 수를 구하고, 예측창에 다시 산 고객인지 확인")
    snippets = [
        (
            "Python",
            "python",
            "pandas 의 groupby 가 묶기를 대신함",
            ROOT / "src" / "labels.py",
            "# 관측창의 고객별 요약",
            "return base",
        ),
        (
            "C",
            "c",
            "거래를 정렬해 두고 고객 번호가 바뀌는 곳마다 새 고객으로 묶음",
            ROOT / "c" / "src" / "labels.c",
            "/* 2회 */",
            "free(rows);",
        ),
        (
            "C#",
            "csharp",
            "Dictionary 로 고객 번호를 바로 찾고 HashSet 으로 송장 수를 셈",
            ROOT / "csharp" / "src" / "Churn.Core" / "Labels" / "LabelBuilder.cs",
            "var summaries = new Dictionary",
            "var newLimit",
        ),
    ]
    codes = {}
    language_cards = []
    for title, language, summary, path, start_text, end_text in snippets:
        code = extract(path, start_text, end_text)
        codes[title] = code
        count_text = "코드를 찾지 못함"
        if code is not None:
            count_text = f"{len(code.splitlines())}줄 (주석 포함)"
        language_cards.append(
            f'<div class="stat" style="border-top:4px solid {LANG_COLOR[title]}">'
            f'<div class="stat-label">{lang_chip(title)}</div>'
            f'<div style="padding:8px 0 4px 0">{summary}</div>'
            f'<div class="stat-sub">{count_text}</div></div>'
        )
    stat_grid(language_cards)

    code_tabs = st.tabs([f"{title} 코드" for title, *_ in snippets])
    for code_tab, (title, language, summary, path, start, end) in zip(code_tabs, snippets):
        with code_tab:
            if codes[title] is None:
                st.warning(f"{path.name} 에서 코드를 찾지 못함")
            else:
                st.code(codes[title], language=language, wrap_lines=True)
                st.caption(f"저장소의 {path.relative_to(ROOT).as_posix()} 에서 읽어 옴")
    st.caption("C 는 이 부분 앞에서 정렬이 끝나 있어야 함 (`transactions.c`) · 줄 수는 코드의 모양을 보기 위한 것이며 언어의 우열이 아님")

    # ---- 4. 결과 일치 확인 ----
    st.markdown("#### 4. 세 언어가 같은 결과를 내는지 확인")
    comparison = load_csv(str(DATA / "comparison.csv"), as_text=True)

    # 파이썬과 C# 의 차이를 허용 오차로 나눠 '허용 오차를 몇 % 썼는지' 로 바꿈
    # 100% 를 넘으면 허용 오차 밖이므로 불일치
    ratios = []
    for _, row in comparison.iterrows():
        python_value = float(row["파이썬"])
        csharp_value = float(row["C#"])
        tolerance = float(row["허용 오차"])
        diff = abs(csharp_value - python_value)
        if tolerance > 0:
            ratio = diff / tolerance * 100
        elif diff == 0:
            ratio = 0.0
        else:
            ratio = 999.0
        ratios.append(ratio)
    comparison = comparison.assign(사용률=ratios)
    inside = int((comparison["사용률"] <= 100).sum())
    worst = comparison.loc[comparison["사용률"].idxmax()]

    stat_grid(
        [
            stat_card("허용 오차 안에서 일치", f"{inside} / {len(comparison)}", "파이썬과 C# 비교", GREEN),
            stat_card("가장 큰 차이", f"{worst['사용률']:.0f}%", f"{worst['항목']} · 허용 오차 대비"),
            stat_card("비교 조건", "시험 고객 1,484명", "정답 CSV 의 계수가 덜 수렴해 1명씩 다른 항목이 있음"),
        ]
    )

    bars = (
        alt.Chart(comparison)
        .mark_bar(clip=True)
        .encode(
            y=alt.Y("항목:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("사용률:Q", title="허용 오차 대비 실제 차이 (%) · 100% 를 넘으면 불일치", scale=alt.Scale(domain=[0, 110])),
            color=alt.condition(alt.datum.사용률 > 100, alt.value(RED), alt.value(TEAL)),
            tooltip=["항목", "파이썬", "C", "C#", "허용 오차", alt.Tooltip("사용률:Q", format=".0f")],
        )
    )
    limit = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(strokeDash=[5, 4], color=ORANGE, size=2).encode(x="x:Q")
    st.altair_chart(styled(bars + limit, 560), width="stretch")
    st.caption("막대가 없는 항목은 파이썬과 값이 같음 · 점선 100% 가 허용 오차의 끝")

    with st.expander("항목별 값 보기 (파이썬 / C / C#)"):
        st.dataframe(
            comparison.drop(columns=["사용률"]).rename(columns={"결과": "판정"}),
            hide_index=True,
            width="stretch",
            height="content",
        )
        st.caption("C 와 C# 의 값은 저장소에서 실행해 둔 결과를 옮긴 것이며 이 앱이 실행한 것이 아님")

# ============================================================
# 탭 4. 방법과 한계
# ============================================================
with tab4:
    st.subheader("방법과 한계")
    st.markdown("#### 작업 방식")
    st.write(
        "Python 을 먼저 만들고 그 결과를 정답지로 삼아 C, C# 순서로 구현  \n"
        "구현 순서·설계·코드 초안은 AI(Claude)와 함께 작성 · 실행과 검증은 직접 수행  \n"
        "파이썬 정답지와 하나씩 대조해 어긋난 부분은 `DEVIATIONS.md`, `LIMITS.md` 에 기록"
    )
    st.markdown("#### 한계 · 결과를 읽을 때 전제로 둘 것")
    st.write("이 프로젝트가 **말할 수 있는 것과 말할 수 없는 것** 의 경계 → 아래 5가지를 전제로 결과를 해석")
    html(
        """
        <table class="grid">
          <tr><th style="width:14%">구분</th><th style="width:43%">한계</th><th style="width:43%">결과를 읽을 때의 영향</th></tr>
          <tr><td>데이터</td>
              <td>2009-2011 년 영국 온라인 소매업체의 거래</td>
              <td>최신 소비 행태와 다름 · 숫자보다 파이프라인과 방법론이 산출물</td></tr>
          <tr><td>가정</td>
              <td>쿠폰 반응률·마진·쿠폰 원가는 가정 (유지 고객 매출 중앙값만 실측)</td>
              <td>쿠폰이 이탈을 되돌렸는지는 이 데이터로 알 수 없음 · 실험이 필요</td></tr>
          <tr><td>모델 비교</td>
              <td>시험 고객 1,484명 기준 · AUC 표집 오차 약 0.012</td>
              <td>0.01 안팎의 차이는 판단 불가 · 로지스틱과 MLP 의 우열을 말할 수 없음</td></tr>
          <tr><td>분할</td>
              <td>학습/시험 분할은 파이썬의 무작위 분할 결과를 C, C# 이 읽어 사용</td>
              <td>분할은 파이썬에 의존 · 독립된 재현은 아님</td></tr>
          <tr><td>속도 비교</td>
              <td>이 PC 에서 잰 참고값 · C 는 최적화 없는 기본 빌드, C# 은 Release 빌드</td>
              <td>조건이 달라 언어 간 속도 우열을 말할 수 없음</td></tr>
        </table>
        """
    )
    st.caption("각 한계의 근거와 겪은 과정은 `LIMITS.md` 와 `DEVIATIONS.md` 에 기록")
    st.markdown("#### 더 보기")
    st.markdown(
        f"[README]({REPO_URL}#readme) · "
        f"[LIMITS.md]({REPO_URL}/blob/main/LIMITS.md) (C 에서 부딪힌 한계와 C# 에서 확인한 것) · "
        f"[DEVIATIONS.md]({REPO_URL}/blob/main/DEVIATIONS.md) (가이드·정답지와 달라진 점)"
    )
