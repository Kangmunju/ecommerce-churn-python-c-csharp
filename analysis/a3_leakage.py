"""8장 — 시점 누수: 글자 세 개(obs → df)가 AUC 를 부풀린다.

python analysis/a3_leakage.py

같은 라벨, 같은 모델, 같은 8개 피처. 피처를 만든 '기간'만 다르다.
  - 정직한 버전: 관측창(cutoff 미만)까지만 본다
  - 누수 버전  : cutoff 를 무시하고 데이터 전체로 만든다 (실수를 그대로 재현)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT =Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import labels as L  # noqa: E402
from train import make_logit, split  # noqa: E402

# 데이터 마지막 날의 다음 날
# 2011-12-10
DATA_END = L.LABEL_END

# 두 버전에 공통으로 만들 수 있는 8개
# 단순한 주문 통계를 사용해 정직한 버전과 누수 버전 모두 같은 방식으로 만들 것
# 주문 단위 통계만 사용함
# 최근 90일 같은 복잡한 피처는 빼서 차이가 오직 기간에서만 생기도록 구현할 것
COMMON = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "std_order_value",
    "max_order_value",
    "total_items",
    "tenure",
]


def leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """cutoff 를 무시하고 데이터 전체로 만든 피처. 실수를 그대로 재현."""

    # features.py 와 같은 방식으로 송장 단위로 접음
    # 관측창으로 자르지 않은 df 전체를 사용함
    orders = (
        df.groupby(["customer_id", "invoice"])
        .agg(
            order_date=("invoice_date", "min"),
            order_value=("revenue", "sum"),
            order_items=("quantity", "sum"),
        )
        .reset_index()
    )

    g = orders.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    last = g.order_date.max()
    first = g.order_date.min()
    f["frequency"] = g.size()
    f["monetary"] = g.order_value.sum()
    f["avg_order_value"] = g.order_value.mean()
    f["std_order_value"] = g.order_value.std().fillna(0)
    f["max_order_value"] = g.order_value.max()
    f["total_items"] = g.order_items.sum()

    # 기준 시점인 recency와 tenure를 기준일이 아닌 데이터 끝에서 부터 셈
    # 12월에 알 수 있는 값을 확인함
    f["recency"] = (DATA_END - last).dt.days
    f["tenure"] = (DATA_END - first).dt.days

    return f


def auc(X: pd.DataFrame, y: np.ndarray) -> float:
    Xtr, Xte, ytr, yte = split(X, y)
    return roc_auc_score(yte, make_logit().fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def main() -> int:
    f = F.build()
    y = f["churn"].to_numpy()
    X_honest = f[COMMON]

    df = pd.read_parquet(F.CLEAN)

    # 같은 고객 집합으로 맞춤
    leaky = leaky_features(df).reindex(f.index)
    assert leaky.notna().all().all(), "누수 피처에 결측이 있음"

    rows = [
        {
            "피처를 만든 구간": "관측창까지만 (cutoff 준수)",
            "피처 수": len(COMMON),
            "AUC": auc(X_honest, y),
        },
        {
            "피처를 만든 구간": "데이터 전체 (cutoff 무시)",
            "피처 수": len(COMMON),
            "AUC": auc(leaky[COMMON], y),
        },
    ]
    r = pd.DataFrame(rows)
    gain = r.AUC[1] - r.AUC[0]
    print("=" * 74)
    print("같은 라벨, 같은 모델. 피처를 만든 기간만 다르다.")
    print("=" * 74)
    print(r.round(4).to_string(index=False))
    print(f"\n  → 공짜로 얻은 AUC {gain:+.4f}. 운영에 올리면 이만큼이 그대로 사라진다.")

    cmp = (
        pd.DataFrame(
            {
                "정직한 recency(관측창 기준)": X_honest["recency"],
                "샌 recency(전체 기준)": leaky["recency"],
                "churn": f["churn"],
            }
        )
        .groupby("churn")
        .mean()
        .round(1)
    )

    print("\n[recency 를 이탈 여부별로 쪼개 본다]")
    print(cmp.to_string())
    print(
        "\n  → 유지 고객(churn=0)은 예측창에 구매가 있어 '마지막 구매'가 최근으로 당겨진다."
    )
    print(
        "    이탈 고객(churn=1)은 구매가 없어 마지막 구매일이 그대로다. 사실상 정답을 옮겨 적은 열이다."
    )

    Path("reports").mkdir(exist_ok=True)
    r.to_csv("reports/a3_leakage.csv", index=False, encoding="utf-8")
    cmp.to_csv("reports/a3_recency_by_churn.csv", encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 1. 샌 recency가 곧 정답인 이유
#    이탈의 정의: 기준일 이후 183일(예측창) 동안 구매가 0건
#    유지 고객은 예측창 안에 구매가 있으므로 데이터 끝에서 센 recency가 항상 183일 이하
#    이탈 고객은 마지막 구매가 기준일 이전이므로 데이터 끝에서 센 recency가 항상 183일 이상
#    -> 샌 recency가 183보다 큰지 작은지가 곧 이탈 여부이므로 모델이 이 경계만 찾으면 거의 완벽하게 맞힘
#    확인: 이탈 고객의 recency는 관측창 기준 238.3일, 데이터 끝 기준 421.3일 (정확히 183일 차이)
#    그래서 AUC가 0.7829에서 0.9996으로 올라 +0.2167이 공짜로 생김
#
# 2. a2_label_definition.py 의 B와의 관계 (서로 거울상)
#    a2의 B: 라벨을 피처(recency)로 만듦 -> 라벨이 피처의 함수가 되어 AUC 1.0000
#    a3: 피처(recency)를 라벨의 미래 정보로 만듦 -> 피처가 라벨의 함수가 되어 AUC 0.9996
#    둘 다 피처와 정답이 사실상 같은 정보가 되어 예측이 아니라 복사가 됨
#
# 3. 운영에서는 이 AUC가 사라짐
#    실제 예측 시점(기준일)에는 이후 183일의 구매를 알 수 없음
#    운영에 올리면 이 피처를 만들 수 없거나 값이 달라져서 AUC +0.2167이 그대로 사라짐
#    누수는 에러도 경고도 없이 AUC만 올리므로 피처를 만들 때 기준일 이전 데이터만 쓰는 것이 유일한 방어
#
# 4. 이 실험의 범위
#    같은 라벨, 같은 모델, 같은 8개 피처로 피처를 만든 기간만 바꿈
#    누수 피처 중 tenure는 모든 고객에 183일이 똑같이 더해진 값이라 표준화하면 정보가 변하지 않음
#    frequency, monetary 등은 미래 주문이 더해져 값이 달라지지만 이 파일은 recency만 따로 쪼개 확인함
#    (피처별로 얼마나 샜는지는 측정하지 않음)
# ====================================================================================================
