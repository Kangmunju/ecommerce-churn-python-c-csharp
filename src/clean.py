"""정제 5단계. 무엇을 왜 버렸는지 단계마다 기록한다 (reports/clean_log.csv).

python src/clean.py

핵심: 취소 전표(반품)는 지우지 않고 따로 떼어 피처(n_returns, return_rate)로 쓴다.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB = Path("data/processed/retail.db")
OUT_CLEAN = Path("data/processed/clean.parquet")
OUT_RETURNS = Path("data/processed/returns.parquet")
LOG = Path("reports/clean_log.csv")

# 상품이 아닌 StockCode
# 배송비·수수료·조정·샘플·테스트·기프트카드 등
# isin으로 포함 여부를 확인하기 위해 {}로 집합(set)을 만들어 진행
# 정규식(^\d{5})으로 거르면 85048A 같은 색상 변형 코드까지 날아가므로 명시적 목록으로 제외함
NON_PRODUCT = {
    "POST",
    "DOT",
    "C2",
    "M",
    "m",
    "D",
    "S",
    "BANK CHARGES",
    "B",
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "CRUK",
    "PADS",
    "gift_0001_10",
    "gift_0001_20",
    "gift_0001_30",
    "gift_0001_40",
    "gift_0001_50",
    "gift_0001_60",
    "gift_0001_70",
    "gift_0001_80",
    "gift_0001_90",
    "TEST001",
    "TEST002",
}


# 단계마다 (단계명, 전, 후, 버린 수, 메모)를 기록하고 화면에서 출력함
# steps 리스트에 튜플을 쌓아 두었다가 마지막에 CSV로 저장하고 검산에도 사용할 것
def _log(steps: list, name: str, before: int, after: int, note: str = "") -> None:
    steps.append((name, before, after, before - after, note))
    print(f"  {name:28s} {before:>10,} → {after:>10,}  (-{before - after:,}) {note}")


def clean() -> pd.DataFrame:
    if not DB.exists():
        raise SystemExit(f"{DB} 가 없습니다. 먼저 python src/db.py 를 실행하세요.")
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM transactions", con, parse_dates=["invoice_date"])

    con.close()
    steps: list = []
    print(f"[정제] 시작 {len(df):,} 행\n")

    # 취소 전표를 떼어 냄
    # 취소 전표는 수량이 음수이므로 이 단계를 가장 먼저 해야 이후 과정이 진행됨
    # 지우는 것이 아니라 반품 이력으로 따로 씀
    is_cancel = df["invoice"].str.startswith("C")
    returns = df[is_cancel].copy()

    # 각 단계마다 n0을 먼저 계산해 버리기 전 행수를 저장하고 _log에 넘김
    n0 = len(df)
    df = df[~is_cancel].copy()
    _log(steps, "1. 취소 전표(C) 분리", n0, len(df), f"→ returns {len(returns):,} 행")

    # 수량이 0 이하인 행을 고름
    # 취소가 아닌데 음수가 나오면 재고 조정이나 파손 등이 있는 것으로 파악함
    n0 = len(df)
    df = df[df["quantity"] > 0]
    _log(steps, "2. 수량 <= 0 제거", n0, len(df))

    # 단가가 0 이하인 것을 고름
    # 증정, 오입력, 매출을 왜곡하는 것으로 파악함
    n0 = len(df)
    df = df[df["price"] > 0]
    _log(steps, "3. 단가 <= 0 제거", n0, len(df))

    # 상품이 아닌 코드를 제거함
    # isin(NON_PRODUCT)은 목록에 있는 코드인 행을 True로 만들고 ~로 뒤집어 그 외의 행만 남김
    n0 = len(df)
    df = df[~df["stock_code"].isin(NON_PRODUCT)]
    _log(steps, "4. 비상품 코드 제거", n0, len(df))

    # 고객 식별 불가
    # 이탈의 경우 '고객'의 상태이므로 CustomerID가 없으면 대상이 아닌 것으로 판단함
    # 버리고 나면 계산할 수 없으므로 버리기 전에 행별 매출(rev)을 구하고 포기하는 매출 비율을 기록함
    # lost : customer_id가 결측인 행이 True인 마스크
    # rev[lost] : 그 행들의 매출만 고른 것
    # lost_pct = 결측 행의 매출 합 / 전체 매출 합 x 100
    n0 = len(df)
    rev = df["quantity"] * df["price"]
    lost = df["customer_id"].isna()
    lost_pct = rev[lost].sum() / rev.sum() * 100
    df = df[~lost].copy()
    _log(steps, "5. CustomerID 결측 제거", n0, len(df), f"매출 {lost_pct:.1f}% 포기")

    # 파생 열을 만들고 정리함
    # 행마다 매출(수량 x 단가)을 계산해 revenue 열로 추가함
    # 결측 행을 모두 버렸으므로 customer_id를 실수(float)에서 정수로 바꿈
    # 행을 지운 뒤 번호가 중간중간 비어 있으므로 reset_index로 0부터 다시 매김
    df["revenue"] = df["quantity"] * df["price"]
    df["customer_id"] = df["customer_id"].astype(int)
    df = df.reset_index(drop=True)

    # 검산 : 최종 = 시작 - 각 단계 감소분
    start = steps[0][1]
    dropped_total = 0
    for s in steps:
        dropped_total += s[3]
    assert start - dropped_total == len(df), "단계별 행 수 합이 맞지 않음"

    LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(steps, columns=["step", "before", "after", "dropped", "note"]).to_csv(
        LOG, index=False, encoding="utf-8"
    )
    df.to_parquet(OUT_CLEAN, index=False)
    returns.reset_index(drop=True).to_parquet(OUT_RETURNS, index=False)

    print(f"\n[정제] 최종 {len(df):,} 행 ({len(df) / start * 100:.1f}% 생존)")
    print(
        f"       고객 {df.customer_id.nunique():,}명 · 송장 {df.invoice.nunique():,}건"
    )
    print(
        f"       기간 {df.invoice_date.min().date()} ~ {df.invoice_date.max().date()}"
    )
    print(f"       매출 £{df.revenue.sum():,.0f}")
    return df


if __name__ == "__main__":
    clean()
    sys.exit(0)
