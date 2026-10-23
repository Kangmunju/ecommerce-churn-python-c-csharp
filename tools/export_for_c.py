"""C·C# 구현이 읽을 입력 파일과, 비교할 정답(파이썬 결과)을 CSV 로 내보낸다.

python tools/export_for_c.py        # 프로젝트 루트에서 실행

입력(C 가 읽는 것) : 정제된 거래 CSV, 반품(취소 전표) CSV
정답(C 가 맞춰 볼 것): 피처 20개, 학습/시험 분할, 모델 계수, 시험셋 예측 확률, 요약 숫자

C 에서는 JSON·parquet 을 읽기 어려우므로 모두 쉼표로 구분한 CSV 로 만든다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
from train import PRED, load_xy, make_logit, split  # noqa: E402

OUT_DIR = Path("data/processed/export_for_c")
PROFIT_CURVE = Path("reports/a5_profit_curve.csv")
ASSUMPTIONS = Path("reports/a5_assumptions.json")

# 날짜는 C에서 직접 해석해야 하는 형태인 'YYYY-MM-DD HH:MM:SS' 문자열로 작성함
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def export_transactions() -> None:
    df = pd.read_parquet(F.CLEAN)

    # C에서 피처를 만들 때 필요한 열만 내보냄
    # 상품 설명(description)은 길어서 C에서 다루기에 번거롭다고 판단함
    # 매출은 수량 * 단가로 C에서 계산 가능해서 제외함
    out = df[
        [
            "customer_id",
            "invoice",
            "stock_code",
            "invoice_date",
            "quantity",
            "price",
            "country",
        ]
    ].copy()

    out["invoice_date"] = out["invoice_date"].dt.strftime(DATE_FMT)
    out.to_csv(OUT_DIR / "clean_transactions.csv", index=False, encoding="utf-8")
    print(f"[입력] clean_transactions.csv {len(out):,}행")


def export_returns() -> None:
    ret = pd.read_parquet(F.RETURNS)

    # 고객 번호가 없는 취소 전표는 피처(n_returns)에 쓰이지 않으므로 내보내지 않음
    ret = ret[ret.customer_id.notna()].copy()
    ret["customer_id"] = ret["customer_id"].astype(int)
    out = ret[["customer_id", "invoice", "invoice_date"]].copy()
    out["invoice_date"] = out["invoice_date"].dt.strftime(DATE_FMT)
    out.to_csv(OUT_DIR / "returns.csv", index=False, encoding="utf-8")
    print(f"[입력] returns.csv             {len(out):,}행")


def export_references() -> None:

    # 피처 20개 + 정답(churn) + 신규 여부(is_new): features.py 가 만든 결과
    f = pd.read_parquet(F.OUT).reset_index()
    f["is_new"] = f["is_new"].astype(int)
    f.to_csv(OUT_DIR / "reference_features.csv", index=False, encoding="utf-8")
    print(f"[정답] reference_features.csv  {len(f):,}행")

    # 학습/시험 분할
    # sklearn 의 층화 분할을 C에서 똑같이 만들 수 없기 때문에 고객 번호를 그대로 넘김
    X, y = load_xy()
    Xtr, Xte, ytr, yte = split(X, y)
    ids = list(Xtr.index) + list(Xte.index)
    labels = ["train"] * len(Xtr) + ["test"] * len(Xte)
    sp = pd.DataFrame({"customer_id": ids, "split": labels})
    sp = sp.sort_values("customer_id")
    sp.to_csv(OUT_DIR / "reference_split.csv", index=False, encoding="utf-8")
    print(f"[정답] reference_split.csv     train {len(Xtr):,} / test {len(Xte):,}")

    # 로지스틱 회귀를 다시 학습해 표준화 값과 계수를 꺼냄
    model = make_logit().fit(Xtr, ytr)
    scaler = model.named_steps["sc"]
    clf = model.named_steps["clf"]
    coef = pd.DataFrame(
        {
            "feature": list(X.columns),
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "coef": clf.coef_[0],
        }
    )

    coef.to_csv(OUT_DIR / "reference_coefficients.csv", index=False, encoding="utf-8")
    print(f"[정답] reference_coefficients.csv  {len(coef)}행")

    # 시험셋 예측 확률
    # train.py 가 저장한 것과 같은지 확인한 뒤 내보냄
    p = model.predict_proba(Xte)[:, 1]
    saved = pd.read_parquet(PRED).set_index("customer_id").loc[Xte.index, "p_logit"]
    assert np.allclose(p, saved.to_numpy()), "train.py 의 예측 확률과 다름"
    pred = pd.DataFrame({"customer_id": Xte.index, "y": yte, "p_logit": p})
    pred.to_csv(OUT_DIR / "reference_test_pred.csv", index=False, encoding="utf-8")
    print(f"[정답] reference_test_pred.csv {len(pred):,}행")

    # C가 마지막에 맞춰 볼 값들을 요약해서 정리함
    pc = pd.read_csv(PROFIT_CURVE)
    best = pc.loc[pc["기대이익"].idxmax()]
    base = pc.loc[pc["임계값"] == 0.50].iloc[0]
    with open(ASSUMPTIONS, encoding="utf-8") as fp:
        a = json.load(fp)

    summary = [
        ("test_auc", roc_auc_score(yte, p)),
        ("intercept", float(clf.intercept_[0])),
        ("n_train", len(Xtr)),
        ("n_test", len(Xte)),
        ("churn_rate_train", float(ytr.mean())),
        ("churn_rate_test", float(yte.mean())),
        ("margin", a["MARGIN"]),
        ("coupon_cost", a["COUPON_COST"]),
        ("response", a["RESPONSE"]),
        ("r_retained_median", a["R_RETAINED_median_measured"]),
        ("value_tp", a["VALUE_TP"]),
        ("value_fp", a["VALUE_FP"]),
        ("best_threshold", float(best["임계값"])),
        ("best_profit", float(best["기대이익"])),
        ("profit_at_0.50", float(base["기대이익"])),
    ]

    pd.DataFrame(summary, columns=["key", "value"]).to_csv(
        OUT_DIR / "reference_summary.csv", index=False, encoding="utf-8"
    )

    print(f"[정답] reference_summary.csv   {len(summary)}개 값")


def main() -> int:
    needed = (F.CLEAN, F.RETURNS, F.OUT, PRED, PROFIT_CURVE, ASSUMPTIONS)

    for p in needed:
        if not p.exists():
            raise SystemExit(
                f"{p} 가 없습니다. 먼저 python run_all.py 로 앞 단계를 실행하세요."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_transactions()
    export_returns()
    export_references()
    print(f"\n[ok  ] {OUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
