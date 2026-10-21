"""이탈 라벨 — 관측창(피처)과 예측창(정답)을 나눈다.

python src/labels.py

    ├──────── 관측창 (피처) ────────┤├─── 예측창 (정답) ───┤
    2009-12-01                 2011-06-10            2011-12-09

운영 시점을 그대로 흉내 낸다: 6월 10일 아침에 아는 것은 그날까지의 이력뿐이고,
궁금한 것은 앞으로 6개월 안에 다시 올지다.
"""

import sys
from pathlib import Path

import pandas as pd

CLEAN = Path("data/processed/clean.parquet")
OUT = Path("data/processed/labels.parquet")

# 예측창 시작 기준일 (관측창은 이 시각 '미만')
CUTOFF = pd.Timestamp("2011-06-10")

# 데이터 마지막 날의 다음 날
# 데이터의 마지막 거래인 2011-12-09로 날짜를 끊으면 9일 전체가 제외됨
# 마지막 날의 다음 날 0시를 끝으로 잡고 <로 비교해 9일 전체가 포함되도록 함
LABEL_END = pd.Timestamp("2011-12-10")

# 관측창 끝 30일 안에 첫 구매한 고객 = 신규
NEW_CUSTOMER_DAYS = 30


def build(
    cutoff: pd.Timestamp = CUTOFF, label_end: pd.Timestamp = LABEL_END
) -> pd.DataFrame:
    """cutoff 를 인자로 제외한 이유: 7장에서 값을 바꿔 가며 실험한다.

    부등호 주의 - 관측창은 '< cutoff', 예측창은 '>= cutoff'. 한쪽만 등호여야 한다.
    """

    df = pd.read_parquet(CLEAN)

    # 관측창 = 기준일 미만
    # 예측창 = 기준일 이상 ~ 마지막 날 다음 날 미만
    obs = df[df.invoice_date < cutoff]
    fut = df[(df.invoice_date >= cutoff) & (df.invoice_date < label_end)]

    # 관측창의 고객별 요약 (첫 구매, 마지막 구매, 주문 수)
    # base는 관측창에서 groupby한 결과이므로 관측창에 1건 이상 구매한 고객만 들어 있음
    # 기준일 이후에 처음 온 고객은 과거 이력이 없어 예측할 수 없으므로 대상에서 빠짐
    # .agg(이름=(열, 함수))는 이 열에 이 함수를 적용해 이 이름으로 저장하라는 문법
    # 고객마다 첫 구매(min), 마지막 구매(max), 서로 다른 송장 수(nunique)를 한 번에 구함
    base = obs.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        n_orders_obs=("invoice", "nunique"),
    )

    # 관측창에 있던 고객이 예측창에도 구매했는지 확인함
    # groupby 결과는 customer_id가 행 이름(인덱스)이 됨
    # fut.customer_id.unique()는 예측창에 구매한 고객 번호 목록
    # isin은 인덱스 각각이 그 목록에 있는지 True/False로 알려줌
    base["bought_in_future"] = base.index.isin(fut.customer_id.unique())

    # 다시 오지 않은 고객을 이탈로 파악
    # ~로 True와 False를 뒤집고 0과 1(int8)로 바꿔 표기함
    base["churn"] = (~base["bought_in_future"]).astype("int8")

    # 관측창 끝 30일 안에 첫 구매한 경우 신규 고객으로 파악
    base["is_new"] = base.first_purchase >= cutoff - pd.Timedelta(
        days=NEW_CUSTOMER_DAYS
    )
    return base


def main() -> int:
    if not CLEAN.exists():
        raise SystemExit(
            f"{CLEAN} 가 없습니다. 먼저 python src/clean.py 를 실행하세요."
        )

    # 날짜 확인에 필요한 두 열만 읽도록 함
    df = pd.read_parquet(CLEAN, columns=["customer_id", "invoice_date"])
    data_end = df.invoice_date.max()

    # 데이터가 LABEL_END 이후까지 있으면 예측창이 데이터를 자르게 되므로 중단함
    if data_end >= LABEL_END:
        raise SystemExit(f"데이터가 LABEL_END({LABEL_END})를 넘어 끝납니다: {data_end}")
    base = build()

    # 요약 숫자 (대상 고객 수, 이탈 수, 신규 고객)
    n = len(base)
    n_churn = int(base.churn.sum())
    new = base[base.is_new]

    # 날짜는 .date()로 시각을 떼고 날짜만 보이도록 함
    # 관측창 끝은 기준일 하루 전이므로 CUTOFF - Timedelta(days=1)로 계산함
    # 관측창 길이는 normalize()(시각을 0시로)한 첫 거래일부터 기준일까지의 일수
    print(f"[라벨] 기준일(cutoff) {CUTOFF.date()}")
    print(
        f"       관측창  {df.invoice_date.min().date()} ~ {(CUTOFF - pd.Timedelta(days=1)).date()}"
        f"  ({(CUTOFF - df.invoice_date.min().normalize()).days}일)"
    )
    print(
        f"       예측창  {CUTOFF.date()} ~ {data_end.date()}"
        f"  ({(LABEL_END - CUTOFF).days}일)"
    )
    print(f"\n       대상 고객  {n:,}명 (관측창에 1건 이상 구매)")
    print(f"       이탈(재구매 없음) {n_churn:,}명  = {n_churn / n * 100:.1f}%")
    print(f"       유지(재구매 있음) {n - n_churn:,}명")
    print(
        f"\n       그중 신규 고객 {len(new):,}명 (관측창 마지막 {NEW_CUSTOMER_DAYS}일 안에 첫 구매)"
    )
    # new.churn.mean()은 0과 1의 평균이므로 이탈률이 됨
    print(
        f"       신규의 이탈률 {new.churn.mean() * 100:.1f}%   vs 기존 {base[~base.is_new].churn.mean() * 100:.1f}%"
    )
    all_c = df.customer_id.nunique()
    print(
        f"\n       정제 후 고객 {all_c:,}명 중 대상 {n:,}명 → 제외 {all_c - n:,}명 "
        f"(기준일 이후에 처음 온 고객, 관측 이력 없음)"
    )

    # 이 표는 customer_id가 인덱스에 들어 있으므로 index=False를 쓰지 않고 인덱스까지 저장함
    # src/build_db.py, src/clean.py에서는 index=False로 저장했으므로 다른 점이니 주의
    base.to_parquet(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
