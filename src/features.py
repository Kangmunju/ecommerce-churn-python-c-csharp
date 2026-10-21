"""고객 피처 20개 — 관측창 밖은 한 줄도 보지 않는다.

python src/features.py

규칙: 이 파일에서 cutoff 이후 데이터는 단 한 줄도 읽지 않는다.
함수 맨 위에서 `obs` 를 만들고, 그 아래로는 원본 `df` 를 쓰지 않는다.
(누수는 에러도 경고도 내지 않고 AUC 만 올린다. 8장에서 그 대가를 측정한다.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 같은 폴더의 labels.py 를 어디서 실행해도 불러올 수 있도록 이 파일의 폴더를 검색 경로에 추가함
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

CLEAN = Path("data/processed/clean.parquet")
RETURNS = Path("data/processed/returns.parquet")
OUT = Path("data/processed/features.parquet")

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "std_order_value",
    "max_order_value",
    "total_items",
    "tenure",
    "active_span",
    "gap_mean",
    "gap_std",
    "gap_max",
    "recency_ratio",
    "orders_90d",
    "revenue_90d",
    "share_90d",
    "n_products",
    "n_returns",
    "return_rate",
    "is_uk",
]


def build(
    cutoff: pd.Timestamp = L.CUTOFF, label_end: pd.Timestamp = L.LABEL_END
) -> pd.DataFrame:
    """고객 1명 = 1행. 피처 20개 + 정답(churn) + 신규 여부(is_new)."""
    df = pd.read_parquet(CLEAN)

    # 관측창 밖은 보지 않음
    obs = df[df.invoice_date < cutoff].copy()

    # 주문 횟수를 행 수로 세면 부풀려지므로 주문 단위는 송장으로 함
    # 상품 한 줄이 아니라 송장 단위로 먼저 접음
    orders = (
        obs.groupby(["customer_id", "invoice"])
        .agg(
            order_date=("invoice_date", "min"),
            order_value=("revenue", "sum"),
            order_items=("quantity", "sum"),
        )
        .reset_index()
        .sort_values(["customer_id", "order_date"])
    )

    g = orders.groupby("customer_id")

    f = pd.DataFrame(index=g.size().index)
    f.index.name = "customer_id"
    first = g.order_date.min()
    last = g.order_date.max()

    # RFM 9개
    # recency: 기준일에서 마지막 주문일을 뺀 일수 -> 이탈 예측에서 가장 강한 신호 중 하나로 예측
    f["recency"] = (cutoff - last).dt.days

    # frequency: 주문 횟수
    f["frequency"] = g.size()

    # monetary: 주문 금액의 합(총 구매액)
    f["monetary"] = g.order_value.sum()

    # 주문 금액의 평균
    f["avg_order_value"] = g.order_value.mean()

    # 주문 금액의 표준편차
    f["std_order_value"] = g.order_value.std().fillna(0)

    # 주문 금액의 최댓값
    f["max_order_value"] = g.order_value.max()

    # 구매 수량의 합
    f["total_items"] = g.order_items.sum()

    # tenure:기준일에서 첫 주문일을 뺀 일수로 고객이 된 지 얼마나 됐는지를 구함
    f["tenure"] = (cutoff - first).dt.days

    # active_span: 첫 주문부터 마지막 주문까지의 기간
    f["active_span"] = (last - first).dt.days

    # 주문 간격으로 개인의 리듬을 파악함
    orders["gap"] = orders.groupby("customer_id").order_date.diff().dt.days
    gg = orders.groupby("customer_id").gap
    f["gap_mean"] = gg.mean()
    f["gap_std"] = gg.std()
    f["gap_max"] = gg.max()

    # 평균으로 채우는 경우 없는 정보를 지어내는 것이 됨
    # 0으로 채우는 경우 비율이 무한대가 되는 것을 확인함
    # 1회 구매 고객은 간격이 없으므로 지금까지 기다린 기간(tenure)을 간격으로 잡음(보수적)
    # 자세한 설명은 파일 하단 [설계 메모] 2 참고
    f["gap_mean"] = f["gap_mean"].fillna(f["tenure"])
    f["gap_std"] = f["gap_std"].fillna(0)
    f["gap_max"] = f["gap_max"].fillna(f["tenure"])

    # 이 사람 리듬 대비 얼마나 늦었는지 확인 (간격 0 이면 '늦지 않았다' = 1.0)
    # 자세한 설명은 파일 하단 [설계 메모] 1 참고
    ratio = f["recency"] / f["gap_mean"].replace(0, np.nan)
    f["recency_ratio"] = ratio.fillna(1.0).clip(upper=50)

    # 90일은 약 한 분기이고 2회 이상 구매 고객의 평균 주문 간격(84일)과 비슷한 길이
    # 평소 리듬대로 사는 고객이라면 최근 90일 안에 한 번쯤은 주문이 있어야 함
    # 따라서 90일을 최근 활동을 보는 기준으로 정함
    recent = orders[orders.order_date >= cutoff - pd.Timedelta(days=90)]
    rg = recent.groupby("customer_id")
    f["orders_90d"] = rg.size().reindex(f.index).fillna(0)
    f["revenue_90d"] = rg.order_value.sum().reindex(f.index).fillna(0)
    f["share_90d"] = f["revenue_90d"] / f["monetary"]

    # 탐색 폭
    f["n_products"] = obs.groupby("customer_id").stock_code.nunique()

    # clean.py에서 떼어 둔 취소 전표를 반품 이력으로 사용함
    # 관측창 안의 것만 씀
    # 자세한 설명은 파일 하단 [설계 메모] 3 참고
    ret = pd.read_parquet(RETURNS)

    # ret.customer_id.notna()로 고객 번호가 없는 행을 거름
    # astype(int)로 정수로 바꾼 뒤 고객별 취소 송장 수(nunique)를 셈
    ret = ret[(ret.invoice_date < cutoff) & ret.customer_id.notna()]
    rc = ret.groupby(ret.customer_id.astype(int)).invoice.nunique()
    f["n_returns"] = rc.reindex(f.index).fillna(0)
    f["return_rate"] = (f["n_returns"] / f["frequency"]).clip(upper=1)

    # 국가의 92%가 영국인 것을 확인함
    # 더미로 다 펴게되면 희소해지므로 is_uk 하나로 접는 것을 선택함
    country = obs.sort_values("invoice_date").groupby("customer_id").country.last()
    f["is_uk"] = (country.reindex(f.index) == "United Kingdom").astype(int)

    f = f[FEATURES].astype("float64")
    assert f.index.is_unique, "고객 인덱스가 중복"

    lab = L.build(cutoff, label_end)[["churn", "is_new"]]

    # how="inner"로 고객 번호(인덱스)가 일치하는 행끼리 붙여줌
    f = f.join(lab, how="inner")
    assert len(f) == len(lab) == len(g.size()), "피처와 라벨의 고객 집합이 다름"
    return f


def main() -> int:
    for p in (CLEAN, RETURNS, L.OUT):
        if not p.exists():
            raise SystemExit(
                f"{p} 가 없습니다. 앞 단계(clean.py, labels.py)를 먼저 실행하세요."
            )
    f = build()
    print(f"[피처] {len(f):,}명 × {len(FEATURES)}개 피처")

    # 표 전체의 결측 개수를 구함
    # 0이 아닌 경우 assert로 멈추도록 작성함
    nan = int(f.isna().sum().sum())
    print(f"       결측 {nan}개")
    assert nan == 0, "피처에 결측이 있다"

    show = [
        "recency",
        "frequency",
        "monetary",
        "gap_mean",
        "recency_ratio",
        "orders_90d",
    ]
    pd.set_option("display.width", 140)
    print("\n       주요 피처 요약")
    print(f[show].describe().round(2).to_string())
    print("\n       이탈 여부별 평균")
    print(f.groupby("churn")[show].mean().round(2).to_string())

    f.to_parquet(OUT)
    print(f"\n[ok  ] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 공통 원칙
# 모든 피처에 "6월 10일 아침에 계산할 수 있는 값인가?"를 물어봄
# 누수는 에러도 경고도 내지 않고 AUC만 올림 (그 대가는 8장에서 측정함)
#
# 1. recency_ratio: 평소 리듬 대비 얼마나 늦었는가
#    예: 둘 다 마지막 주문 후 60일이 지난 고객
#        A는 평소 30일 간격 -> 60 / 30 = 2.0 (늦음)
#        B는 평소 365일 간격 -> 60 / 365 = 0.16 (정상)
#    recency만 보면 둘이 구분되지 않으므로 비율로 그 사람 기준의 지연을 봄
#    recency와 gap_mean 모두 관측창 안의 주문에서만 나오므로 누수가 없음
#    gap_mean을 기준일 이후 주문까지 넣어 계산하면 정답이 피처에 스며들게 됨
#
# 2. 1회 구매 고객 처리
#    주문이 1건이면 직전 주문이 없어 diff()가 NaN이 되므로 무언가로 채워야 함
#    0으로 채우면 비율이 무한대가 되거나 replace를 거쳐 '늦지 않았다'(1.0)가 되어 사실과 다름
#    전체 평균으로 채우면 그 고객에 대해 모르는 리듬을 지어내는 것이 됨
#    tenure로 채우면 '지금까지 기다린 만큼이 간격'이라는 보수적인 가정이 됨
#    첫 주문일 = 마지막 주문일이라 recency = tenure가 되어 recency_ratio는 항상 1.0(중립)
#    이 고객들은 recency, frequency 등 다른 피처로 판단하게 됨
#    누수: 기준일 이후 주문까지 보고 간격을 계산하면 간격이 있는지 없는지가 곧 재구매 여부(정답)가 됨
#
# 3. 반품은 관측창 안의 것만 사용
#    returns.parquet에는 전체 기간의 취소 전표가 들어 있음
#    기준일 이후 반품은 6월 10일 아침에 알 수 없는 정보임
#    반품은 가게와 접촉했다는 신호라 이탈과 반대 방향으로 강하게 연결되어 AUC가 부풀려짐
#    clean.parquet(주문)과 다른 파일에서 읽는 데이터라 obs 필터를 빼먹기 쉬우므로 invoice_date < cutoff를 명시함
# ====================================================================================================
