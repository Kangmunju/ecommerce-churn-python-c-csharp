"""parquet → SQLite 적재. 중복은 파이썬이 아니라 DB 의 UNIQUE 제약이 막는다.

python src/db.py            # 적재 + 검증 (몇 번을 돌려도 결과가 같다 = 멱등)
python src/db.py --reset    # DB 를 지우고 처음부터

실측 (2026-10-21, 원본 UCI Online Retail II 기준)
  입력 1,067,371행 → 저장 1,033,034행 · UNIQUE 로 막힘 34,337행
  (시트 안 12,135 + 시트 사이 22,202)
  두 번째 실행은 저장 0행 · 적재 이력만 +1 → 멱등
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

SRC = Path("data/raw/retail_raw.parquet")
DB = Path("data/processed/retail.db")

# 테이블 정의 / 같은 5개 열 조합이 이미 있으면 UNIQUE 가 막음
# CREATE TABLE IF NOT EXISTS로 표가 이미 있으면 다시 만들지 않도록 함
# load_log 표에 적재할 때마다 입력/저장/막힌 행 수를 한 줄씩 기록함
SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    invoice       TEXT    NOT NULL,
    stock_code    TEXT    NOT NULL,
    description   TEXT,
    quantity      INTEGER NOT NULL,
    invoice_date  TEXT    NOT NULL,
    price         REAL    NOT NULL,
    customer_id   INTEGER,
    country       TEXT,
    UNIQUE (invoice, stock_code, quantity, price, invoice_date)
);
CREATE INDEX IF NOT EXISTS ix_tx_customer ON transactions (customer_id);
CREATE INDEX IF NOT EXISTS ix_tx_date     ON transactions (invoice_date);

CREATE TABLE IF NOT EXISTS load_log (
    loaded_at    TEXT,
    rows_in      INTEGER,
    rows_stored  INTEGER,
    rows_blocked INTEGER
);
"""

# 적재할 열 (INSERT 의 열 순서와 같아야 함)
COLS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "CustomerID",
    "Country",
]
# UNIQUE 키 (build_db.py 와 동일)
KEY = ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]


def load() -> None:
    df = pd.read_parquet(SRC)
    out = df[COLS].copy()

    # SQLite에는 날짜 타입이 존재하지 않아 문자열로 저장함
    out["InvoiceDate"] = out["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # sqlite3는 pandas의 pd.NA를 모르므로 None으로 바꿔주어야 NULL로 들어감
    # notna()로 True 값이 있는 표를 만들어줌
    # where(조건, None)으로 조건이 False인 결측칸을 None으로 바꾸어 줌
    # itertuples(name=None)으로 행을 일반 튜플로 꺼내 executemany가 받도록 함
    rows = list(
        out.astype(object).where(out.notna(), None).itertuples(index=False, name=None)
    )

    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    before = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    con.executemany(
        "INSERT OR IGNORE INTO transactions "
        "(invoice, stock_code, description, quantity, invoice_date, price, "
        "customer_id, country) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    # 새로 저장된 행 = 적재 후 - 적재 전
    # 막힌 행 = 입력 - 저장
    stored = after - before
    blocked = len(rows) - stored
    con.execute(
        "INSERT INTO load_log VALUES (datetime('now'), ?, ?, ?)",
        (len(rows), stored, blocked),
    )

    # 적재 결과와 로그를 따로 확정함
    con.commit()
    con.close()
    print(f"[load] 입력 {len(rows):,} 행")
    print(f"       저장 {stored:,} 행 · UNIQUE 로 막힘 {blocked:,} 행")
    print(f"       테이블 총 {after:,} 행")


def duplicate_breakdown() -> None:
    """막힌 중복이 '시트 안' 인지 '시트 사이' 인지 갈라서 센다 (원인이 다름)."""
    df = pd.read_parquet(SRC, columns=KEY + ["source_sheet"])
    total = int(df.duplicated(KEY).sum())

    # 시트별로 따로 중복을 세어 더하는 것으로 시트 안의 중복을 찾도록 함
    within = 0
    for _, g in df.groupby("source_sheet"):
        within += int(g.duplicated(KEY).sum())
    print("\n[중복 분해]")
    print(f"  UNIQUE 가 막은 총 초과행 : {total:,}")
    print(f"    시트 '안'의 중복       : {within:,}")
    print(
        f"    시트 '사이'의 중복     : {total - within:,}   ← 2010-12-01~12-09 가 두 시트에 다 들어 있음"
    )


# 적재가 끝난 뒤 DB에서 직접 행 수, 고객 수, 송장 수, 기간을 다시 조회해서 확인
# load_log의 줄 수인 적재 이력을 출력해 몇 번 실행했는지 알 수 있도록 함
def verify() -> None:
    con = sqlite3.connect(DB)
    n, cust, inv, lo, hi = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT customer_id), COUNT(DISTINCT invoice), "
        "MIN(invoice_date), MAX(invoice_date) FROM transactions"
    ).fetchone()

    runs = con.execute("SELECT COUNT(*) FROM load_log").fetchone()[0]

    con.close()
    print("\n[검증]")
    print(f"  행 수      {n:>10,}")
    print(f"  고객 수    {cust:>10,}")
    print(f"  송장 수    {inv:>10,}")
    print(f"  기간       {lo} ~ {hi}")
    print(f"  적재 이력  {runs:>10} 회")


# 적재 -> 검증 -> 중복 분해
def main() -> int:
    if not SRC.exists():
        raise SystemExit(
            f"{SRC} 가 없습니다. 먼저 python src/build_db.py 를 실행하세요."
        )
    DB.parent.mkdir(parents=True, exist_ok=True)
    # --reset 이 있으면 DB 파일을 지우고 처음부터 적재함
    if "--reset" in sys.argv and DB.exists():
        DB.unlink()
        print(f"[reset] {DB} 삭제")
    load()
    verify()
    duplicate_breakdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
