"""45MB 엑셀을 한 번만 읽어 parquet 으로 굳힌다. (엑셀을 읽는 곳은 프로젝트에서 여기뿐이다)

python src/build_db.py

dtype 지정이 핵심이다. 안 하면 Invoice('C489449' 같은 취소 전표)와 Description(숫자만 든 칸)
때문에 parquet 저장에서 죽는다.
"""

import sys
import time
from pathlib import Path

import pandas as pd

XLSX = Path("data/raw/online_retail_II.xlsx")
OUT = Path("data/raw/retail_raw.parquet")
TEXT_COLS = ["Invoice", "StockCode", "Description", "Country"]

# UNIQUE 키 (db.py 와 동일)
KEY = ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]


def load_excel() -> pd.DataFrame:
    xl = pd.ExcelFile(XLSX)
    print(f"[read] 시트 {xl.sheet_names}")
    frames = []
    for sheet in xl.sheet_names:
        # 문자로 읽어야 하는 열을 처음부터 str로 지정함(섞인 타입 방지)
        d = xl.parse(
            sheet, dtype={"Invoice": str, "StockCode": str, "Description": str}
        )
        # 어느 시트에서 온 행인지 남김(겹치는 기간 점검에 사용)
        d["source_sheet"] = sheet
        print(f"        {sheet:16s} {len(d):>9,} 행")
        frames.append(d)
    # 시트가 여러 개라서 세로로 이어 붙임
    df = pd.concat(frames, ignore_index=True)
    new_cols = []
    for c in df.columns:
        new_cols.append(c.replace(" ", ""))
    df.columns = new_cols
    for c in TEXT_COLS:
        df[c] = df[c].astype("string")
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    return df


# 두 시트가 겹치지 않을 것으로 예상했으나 실제로 9일이 겹치는 것을 확인함
# 중복을 지우지 않고 존재만 확인해서 출력할 것
# 실제 제거는 clean.py에서 하도록 작성함
def overlap_report(df: pd.DataFrame) -> None:
    print("\n[시트 기간 점검]")
    spans = {}
    for s, g in df.groupby("source_sheet"):
        spans[s] = (g.InvoiceDate.min(), g.InvoiceDate.max())
        print(
            f"      {s}: {g.InvoiceDate.min()} ~ {g.InvoiceDate.max()} ({len(g):,}행)"
        )
    (a0, a1), (b0, b1) = spans.values()

    # 겹치는 구간 = 늦은 시작 ~ 이른 끝
    lo, hi = max(a0, b0), min(a1, b1)
    if lo > hi:
        print("       겹치는 기간 없음")
        return

    # 그날 하루를 통째로 포함
    # inclusive="left"로 마지막을 제외함
    # 끝 날짜의 다음 날 0시
    hi_day = hi.normalize() + pd.Timedelta(days=1)
    m = df.InvoiceDate.between(lo.normalize(), hi_day, inclusive="left")
    print(f"\n        겹치는 기간({lo.date()} ~ {hi.date()}) 행수 {int(m.sum())}")
    inv = df[m].groupby("source_sheet").Invoice.nunique()

    # 두 시트 모두에 등장하는 송장은 집합의 교집합으로 파악
    sheet_invoices = []
    for _, g in df[m].groupby("source_sheet"):
        sheet_invoices.append(set(g.Invoice))
    # *는 리스트를 풀어서 각각을 별개의 인자로 넘김
    # sheet_invoices = [A, B]일 때 set.intersection(*sheet_invoices) == set.intersection(A, B)
    # 시트 수가 몇 개든 같은 코드로 교집합을 구할 수 있음
    both = set.intersection(*sheet_invoices)
    print(f"      송장수 {inv.to_dict()} / 양쪽에 모두 있는 송장 {len(both)}")
    rev = (df[m].Quantity * df[m].Price).sum()
    print(f"      겹침 구간 매출(중복 포함) £{rev:,.0f} → 절반 £{rev / 2:,.0f} 이 허수")


# 엑셀 읽기 -> 겹침 점검 -> parquet 저장순서로 실행
def main() -> int:
    t = time.time()
    if not XLSX.exists():
        raise SystemExit(
            f"{XLSX}가 없습니다. 먼저 python src/download_data.py를 실행하세요"
        )
    df = load_excel()
    overlap_report(df)

    # index=False를 이용해 행 번호를 파일에 저장하지 않음
    df.to_parquet(OUT, index=False)
    print(f"\n[ok ] {OUT} {len(df):,}행 × {df.shape[1]} 열  ({time.time() - t:.1f}초)")
    print(f"       기간 {df.InvoiceDate.min()} ~ {df.InvoiceDate.max()}")
    print(
        f"       고객 {df.CustomerID.nunique():,}명 · 송장 {df.Invoice.nunique():,}건 · "
        f"상품 {df.StockCode.nunique():,}종"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
