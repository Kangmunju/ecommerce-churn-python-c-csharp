"""7장 — 이탈의 '정의'가 성능을 만든다.

python analysis/a2_label_definition.py

데이터·피처·모델을 고정하고 정의만 바꾼다.
  실험 1: 예측창 길이 (30/60/90/183/270일)
  실험 2: 정의 방식 — 미래를 보는가(A) / 과거만 보는가(B), 그리고 피처를 빼면(C)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 프로젝트 루트(analysis/ 의 한 단계 위)를 구하고 src/ 를 모듈 검색 경로에 추가함
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
import labels as L  # noqa: E402
from train import make_logit, split  # noqa: E402

# 데이터 마지막 날의 다음 날 (2011-12-10)
DATA_END = L.LABEL_END

# 입력(X)에서 빼는 열
CUSTOMER_COL_DROP = ["churn", "is_new"]


# 같은 모델(로지스틱)과 같은 분할로 AUC만 계산함
# 모델과 분할이 같으면 AUC 차이는 정의를 바꾼 데서만 생김
def fit_auc(X: pd.DataFrame, y: np.ndarray) -> float:
    Xtr, Xte, ytr, yte = split(X, y)
    m = make_logit().fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def main() -> int:
    Path("reports").mkdir(exist_ok=True)

    # 실험 1: 예측창 길이를 바꿔 가며 비교함
    # 데이터 끝이 고정이므로 데이터 끝에서 예측창 길이만큼 뺀 날을 기준일로 함
    rows = []
    for days in (30, 60, 90, 183, 270):
        cutoff = DATA_END - pd.Timedelta(days=days)

        # 기준일이 바뀌면 관측창도 바뀌므로 피처와 라벨을 다시 만듦
        f = F.build(cutoff=cutoff, label_end=DATA_END)
        X, y = f.drop(columns=CUSTOMER_COL_DROP), f["churn"].to_numpy()
        rows.append(
            {
                "예측창(일)": days,
                "기준일": cutoff.date(),
                "대상 고객": len(f),
                "이탈률": y.mean(),
                "AUC": fit_auc(X, y),
            }
        )

    w = pd.DataFrame(rows)
    print("[실험 1 - 예측창 길이]")
    print(w.round(4).to_string(index=False))
    w.to_csv("reports/a2_window_length.csv", index=False, encoding="utf-8")

    # 예측창은 고객의 평소 구매 간격보다 길어야 함
    # 2회 이상 구매 고객의 실제 간격은 84일이므로 그 약 2배인 183일로 정함
    f183 = F.build()
    gap_all = f183.gap_mean.mean()
    gap_rep = f183.loc[f183.frequency >= 2, "gap_mean"].mean()
    print(
        f"\n 평소 구매 간격: gap_mean 평균 {gap_all:.0f}일 (1회 구매 고객은 tenure 로 채움)"
        f" / 2회 이상 고객만 {gap_rep:.0f}일"
    )

    # 실험 2: 라벨을 만드는 방식(A, B, C)을 바꿔 가며 비교함
    X = f183.drop(columns=CUSTOMER_COL_DROP)
    recency_family = [
        "recency",
        "recency_ratio",
        "gap_max",
        "active_span",
        "tenure",
        "orders_90d",
        "revenue_90d",
        "share_90d",
    ]

    # A: 예측창 안의 구매가 0건이면 이탈로 처리함
    y_win = f183["churn"].to_numpy()
    out = [("A. 예측창 183일 안에 구매 0건", y_win.mean(), fit_auc(X, y_win))]

    # B: 마지막 구매 후 N일이 지났으면 이탈로 처리함 (기준일 이전 데이터만 사용)
    for n in (90, 180):
        y_rec = (X["recency"] >= n).astype(int).to_numpy()
        out.append((f"B. 마지막 구매 후 {n}일 경과", y_rec.mean(), fit_auc(X, y_rec)))

    # C: B(180일)의 라벨은 그대로 두고 recency 계열 피처를 빼고 다시 확인함
    # 라벨은 recency가 180 이상이면 1, 아니면 0인 배열
    y_rec180 = (X["recency"] >= 180).astype(int).to_numpy()
    out.append(
        (
            f"C. B(180일) 인데 recency 계열 피처 {len(recency_family)}개 제거",
            y_rec180.mean(),
            fit_auc(X.drop(columns=recency_family), y_rec180),
        )
    )

    d = pd.DataFrame(out, columns=["정의", "이탈률", "AUC"])
    print("\n[실험 2 - 정의 방식]")
    print(d.round(4).to_string(index=False))
    d.to_csv("reports/a2_definition_type.csv", index=False, encoding="utf-8")

    print("\n→ B 의 AUC 1.0000 은 예측이 아니라 복사다: 라벨이 피처(recency)의 함수다.")
    print(
        "→ C 처럼 열 8개를 빼도 A 보다 높다: 남은 피처도 recency 와 상관돼 있다. 누수는 열을 지워서 안 없어진다."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 1. 기준일을 거꾸로 계산하는 이유
#    보통은 기준일(6월 10일)을 먼저 정하고 그 뒤로 예측창이 정해짐
#    이 실험은 데이터 끝(2011-12-10)이 고정이고 예측창 길이가 바뀌므로
#    기준일 = 데이터 끝 - 예측창 길이로 계산함 (30일 -> 11월 10일, 183일 -> 6월 10일)
#    기준일이 바뀌면 관측창(피처를 만드는 기간)도 바뀌므로 매번 F.build(cutoff=...)로 피처를 다시 만듦
#
# 2. 정의가 다르면 AUC를 서로 비교할 수 없음
#    예측창이 짧을수록 이탈률이 올라가고(30일 74.1% -> 270일 41.6%) AUC는 조금 높음 (0.8032 vs 0.7720)
#    정의가 다르면 문제의 난이도가 달라지므로 AUC가 높은 정의가 좋은 정의라고 할 수 없음
#
# 3. 예측창을 183일로 정한 근거
#    이탈률이 48%로 균형이 잡힘 (30일 74.1%, 270일 41.6%)
#    2회 이상 구매 고객의 평소 주문 간격 84일의 약 2배
#    gap_mean 전체 평균 140일은 1회 구매 고객을 tenure로 채운 값이 끌어올린 것이므로 근거로 쓰지 않음
#
# 4. B가 예측이 아니라 복사인 이유
#    A는 기준일 이후(미래)의 구매 여부로 라벨을 만들어 피처(기준일 이전)와 독립임
#    B는 기준일 이전 데이터(recency)만으로 라벨을 만듦 -> 라벨이 피처 하나의 규칙이 됨
#    모델은 recency를 보고 규칙을 그대로 따라 하면 되므로 AUC가 1.0000(90일), 0.9999(180일)이 됨
#    180일이 정확히 1이 아닌 이유: 로지스틱 회귀는 직선형 경계라 계단형 규칙을 완벽히 표현하지 못함
#
# 5. C: 누수는 열을 지워도 사라지지 않음
#    recency 계열 피처 8개를 빼도 AUC가 0.8281로 A(0.7818)보다 높음
#    남은 피처(gap_mean, 구매 횟수 등)도 recency와 상관이 있어 라벨을 간접적으로 알려 줌
#    정의 자체가 문제이므로 열을 지우는 것이 아니라 정의를 고쳐야 함
# ====================================================================================================
