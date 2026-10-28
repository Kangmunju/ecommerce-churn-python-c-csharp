"""C 구현 시험(c/src/main.c)의 기대값을 파이썬으로 구한다.

    python tools/expected_values.py

c/src/main.c 의 check(...) 마지막 인자가 이 파일의 출력과 같아야 한다.
입력은 tools/export_for_c.py 가 내보낸 CSV 이고, C 프로그램이 읽는 파일과 같다.
시각은 모두 '1970-01-01 00:00:00 부터 지난 초' 로 바꿔서 더한다(C 의 dt_parse 와 같은 기준).
"""

from pathlib import Path


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXPORT = Path("data/processed/export_for_c")
EPOCH = pd.Timestamp("1970-01-01")
CUTOFF = pd.Timestamp("2011-06-10")  # 관측창은 이 시간 '미만'으로 설정함


def to_seconds(text: pd.Series) -> pd.Series:
    """'YYYY-MM-DD HH:MM:SS' 글자를 1970-01-01 부터 지난 초(정수)로 바꾼다."""

    # C의 dt_parse가 하는 일과 동일함
    return (pd.to_datetime(text) - EPOCH) // pd.Timedelta(seconds=1)


# 항목 이름을 28칸 폭으로 맞춰 출력하는 함수를 작성함
def show(name: str, value) -> None:
    print(f"  {name:<28} {value}")


def transactions() -> None:
    """거래 적재: 거래 수, 합계, 첫 행과 마지막 행."""

    df = pd.read_csv(
        EXPORT / "clean_transactions.csv", dtype={"invoice": str, "stock_code": str}
    )
    revenue = df["quantity"] * df["price"]

    print("[clean_transactions.csv]")
    show("거래 수", len(df))
    show("고객 번호 합계", int(df["customer_id"].sum()))
    show("수량 합계", int(df["quantity"].sum()))
    show("초 합계", int(to_seconds(df["invoice_date"]).sum()))
    show("매출 합계(센트)", round(revenue.sum() * 100))
    show("첫 행", df.astype(str).iloc[0].tolist())
    show("마지막 행", df.astype(str).iloc[-1].tolist())


def labels() -> None:
    """이탈 라벨: 파이썬 정답지가 만든 피처 파일의 라벨 열을 집계한다."""

    ref = pd.read_csv(EXPORT / "reference_features.csv")

    print("[reference_features.csv]")
    show("대상 고객", len(ref))
    show("이탈 고객", int(ref["churn"].sum()))
    show("신규 고객", int(ref["is_new"].sum()))
    show("주문 수 합계(frequency)", int(ref["frequency"].sum()))


def returns() -> None:
    """반품 전표: 반품 수, 합계, 기준일 이전 반품 수, 첫 행과 마지막 행."""
    df = pd.read_csv(EXPORT / "returns.csv", dtype={"invoice": str})

    print("[returns.csv]")
    show("반품 수", len(df))
    show("고객 번호 합계", int(df["customer_id"].sum()))
    show("초 합계", int(to_seconds(df["invoice_date"]).sum()))
    before = pd.to_datetime(df["invoice_date"]) < CUTOFF
    show("2011-06-10 이전 반품 수", int(before.sum()))
    show("첫 행", df.astype(str).iloc[0].tolist())
    show("마지막 행", df.astype(str).iloc[-1].tolist())


def logistic() -> None:
    """로지스틱 회귀: 학습 고객으로 끝까지 수렴시킨 계수와 절편."""

    ref = pd.read_csv(EXPORT / "reference_features.csv")
    split = pd.read_csv(EXPORT / "reference_split.csv")
    names = pd.read_csv(EXPORT / "reference_coefficients.csv")["feature"]
    train = ref.merge(split, on="customer_id")
    train = train[train["split"] == "train"]

    z = StandardScaler().fit_transform(train[names])

    # 정답 CSV는 sklearn 기본 허용 오차(1e-4)에서 멈춘 값이라 끝까지 수렴시킨 값을 따로 구함
    model = LogisticRegression(tol=1e-12, max_iter=100000).fit(z, train["churn"])

    print("[로지스틱 회귀 (tol=1e-12)]")
    show("학습 고객", len(train))
    show("반복 횟수", int(model.n_iter_[0]))
    show("절편", f"{model.intercept_[0]:.10f}")
    for name, coef in zip(names, model.coef_[0]):
        show(name, f"{coef:+.10f}")


def main() -> int:
    for name in (
        "clean_transactions.csv",
        "reference_features.csv",
        "reference_split.csv",
        "returns.csv",
    ):
        if not (EXPORT / name).exists():
            raise SystemExit(
                f"{EXPORT / name} 가 없습니다. 먼저 python tools/export_for_c.py 를 실행하세요."
            )
    transactions()
    print()
    labels()
    print()
    returns()
    print()
    logistic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
