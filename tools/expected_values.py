"""C 구현 시험(c/src/main.c)의 기대값을 파이썬으로 구한다.

    python tools/expected_values.py

c/src/main.c 의 check(...) 마지막 인자가 이 파일의 출력과 같아야 한다.
입력은 tools/export_for_c.py 가 내보낸 CSV 이고, C 프로그램이 읽는 파일과 같다.
시각은 모두 '1970-01-01 00:00:00 부터 지난 초' 로 바꿔서 더한다(C 의 dt_parse 와 같은 기준).
"""

from pathlib import Path


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


EXPORT = Path("data/processed/export_for_c")
EPOCH = pd.Timestamp("1970-01-01")
CUTOFF = pd.Timestamp("2011-06-10")  # 관측창은 이 시간 '미만'으로 설정함
LABEL_END = pd.Timestamp("2011-12-10")  # 예측창은 이 시간 '미만'까지로 설정함


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


def metrics() -> None:
    """평가 지표: 정답 CSV 의 시험 고객 확률(p_logit)로 지표 7개와 혼동행렬을 구한다."""

    df = pd.read_csv(EXPORT / "reference_test_pred.csv")
    y = df["y"]
    prob = df["p_logit"]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)

    print("[reference_test_pred.csv, 임계값 0.5]")
    show("고객 수", len(df))
    show("AUC", f"{roc_auc_score(y, prob):.12f}")
    show("PR-AUC", f"{average_precision_score(y, prob):.12f}")
    show("정확도", f"{accuracy_score(y, pred):.12f}")
    show("정밀도", f"{precision_score(y, pred, zero_division=0):.12f}")
    show("재현율", f"{recall_score(y, pred, zero_division=0):.12f}")
    show("F1", f"{f1_score(y, pred, zero_division=0):.12f}")
    show("Brier", f"{brier_score_loss(y, prob):.12f}")
    show("TN FP FN TP", f"{cm[0, 0]} {cm[0, 1]} {cm[1, 0]} {cm[1, 1]}")

    # 확률이 전부 같을 때(동점)도 C 와 같은 값이 나오는지 확인하는 시험용 기대값
    flat = pd.Series([0.5] * len(df))
    print("[확률을 전부 0.5 로 둔 경우]")
    show("AUC", f"{roc_auc_score(y, flat):.12f}")
    show("PR-AUC", f"{average_precision_score(y, flat):.12f}")


def profit() -> None:
    """기대이익: 유지 고객의 예측창 매출 중앙값과 임계값별 이익 곡선."""

    # 가정 (analysis/a5_profit.py 와 같은 값)
    margin, coupon_cost, response = 0.30, 5.00, 0.10

    df = pd.read_csv(
        EXPORT / "clean_transactions.csv", dtype={"invoice": str, "stock_code": str}
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    window = df[(df["invoice_date"] >= CUTOFF) & (df["invoice_date"] < LABEL_END)]
    observed = df[df["invoice_date"] < CUTOFF]["customer_id"].unique()
    revenue = (
        window[window["customer_id"].isin(observed)]
        .groupby("customer_id")["revenue"]
        .sum()
    )

    value_tp = response * revenue.median() * margin - coupon_cost
    value_fp = -coupon_cost

    print("[유지 고객의 예측창 매출]")
    show("고객 수", len(revenue))
    show("중앙값", f"{revenue.median():.10f}")
    show("평균", f"{revenue.mean():.10f}")
    show("맞게 보낸 1건의 이익", f"{value_tp:.10f}")
    show("헛되이 보낸 1건의 손실", f"{value_fp:.10f}")

    test = pd.read_csv(EXPORT / "reference_test_pred.csv")
    y, prob = test["y"], test["p_logit"]

    print("[임계값별 이익 (reference_test_pred.csv)]")
    best_threshold, best_profit = 0.0, None
    total_sent, total_profit = 0, 0.0
    for k in range(5, 100, 5):
        threshold = k / 100.0
        sent = prob >= threshold
        tp = int((sent & (y == 1)).sum())
        fp = int((sent & (y == 0)).sum())
        value = tp * value_tp + fp * value_fp
        total_sent += tp + fp
        total_profit += value
        if best_profit is None or value > best_profit:
            best_threshold, best_profit = threshold, value
        if k in (30, 50):
            show(
                f"임계값 {threshold:.2f}",
                f"발송 {tp + fp}, TP {tp}, FP {fp}, 이익 {value:.6f}",
            )
    show("최적 임계값", f"{best_threshold:.2f}")
    show("최대 이익", f"{best_profit:.6f}")
    show("19개 발송 수 합계", total_sent)
    show("19개 이익 합계", f"{total_profit:.6f}")


def main() -> int:
    for name in (
        "clean_transactions.csv",
        "reference_features.csv",
        "reference_split.csv",
        "reference_test_pred.csv",
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
    print()
    metrics()
    print()
    profit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
