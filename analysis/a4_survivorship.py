"""9장 — 생존편향: "이탈률 48%" 는 누구를 센 숫자인가.

python analysis/a4_survivorship.py

예측창·정의·모델은 그대로 두고 '누구를 모집단에 넣느냐'만 바꾼다.
이탈률은 데이터에서 읽는 숫자가 아니라 모집단을 정한 뒤에 나오는 숫자다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402
from train import SEED, make_logit, split  # noqa: E402


# 같은 분할과 모델로 시험셋 AUC를 한 번 구하는 함수를 작성함
def fit_auc(X: pd.DataFrame, y: np.ndarray) -> float:
    Xtr, Xte, ytr, yte = split(X, y)
    return roc_auc_score(yte, make_logit().fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def main() -> int:
    f = F.build()

    # 누구를 넣을지 모집단을 바꿔가며 비교함
    cases = {
        "전체 (아무것도 빼지 않음)": f,
        "1회 구매 고객 제외": f[f.frequency >= 2],
        "신규(기준일 직전 30일 내 첫 구매) 제외": f[~f.is_new],
        # f[f.frequency >= 2]: 주문 횟수가 2회 이상인 고객만 남김
        # f[~f.is_new]: 신규가 아닌 고객만 남김
        # and 조건으로 묶어줌
        "1회 구매 + 신규 모두 제외": f[(f.frequency >= 2) & (~f.is_new)],
        "2회 이상 + 관측창 180일 이상 지켜본 고객만": f[
            (f.frequency >= 2) & (f.tenure >= 180)
        ],
    }

    rows = []
    for name, sub in cases.items():
        X = sub.drop(columns=["churn", "is_new"])
        y = sub["churn"].to_numpy()

        # 단일 분할 AUC는 모집단이 작을수록 흔들리므로 5-fold CV 평균±표준편차를 같이 냄
        cv = cross_val_score(
            make_logit(),
            X,
            y,
            scoring="roc_auc",
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        )

        rows.append(
            {
                "모집단": name,
                "고객 수": len(sub),
                "이탈률": y.mean(),
                "AUC": fit_auc(X, y),
                "CV AUC": cv.mean(),
                "CV 표준편차": cv.std(),
            }
        )

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(t.round(4).to_string(index=False))
    t.to_csv("reports/a4_survivorship.csv", index=False, encoding="utf-8")

    # 1회 구매 고객과 2회 이상 고객을 나눠서 봄
    one = f[f.frequency == 1]
    rep = f[f.frequency >= 2]

    print("\n[1회 구매 고객을 따로 보면]")
    print(
        f"  1회 구매 고객 {len(one):,}명 ({len(one) / len(f) * 100:.1f}%) · 이탈률 {one.churn.mean() * 100:.1f}%"
    )
    print(
        f"  2회 이상 고객 {len(rep):,}명 ({len(rep) / len(f) * 100:.1f}%) · 이탈률 {rep.churn.mean() * 100:.1f}%"
    )

    seg = pd.DataFrame(
        {
            "집단": ["1회 구매", "2회 이상"],
            "고객 수": [len(one), len(rep)],
            "비중": [len(one) / len(f), len(rep) / len(f)],
            "이탈률": [one.churn.mean(), rep.churn.mean()],
        }
    )
    seg.to_csv("reports/a4_segments.csv", index=False, encoding="utf-8")
    print(
        f"\n  → 전체 이탈률 {f.churn.mean() * 100:.1f}% 는 두 집단의 가중평균일 뿐, 어느 쪽도 대표하지 않는다."
    )

    base_cv = t.loc[0, "CV AUC"]
    worst_sd = t["CV 표준편차"].max()
    print(
        f"  → 단일 분할 AUC 는 모집단에 따라 {t['AUC'].min():.2f}~{t['AUC'].max():.2f} 로 오르내리지만, "
        f"5-fold CV 로 보면 {t['CV AUC'].min():.3f}~{t['CV AUC'].max():.3f} 이고 표준편차가 최대 {worst_sd:.3f} 다."
    )
    print(
        f"    1회 구매 고객을 빼면 단일 분할 AUC 는 오르지만({t.loc[0, 'AUC']:.4f}→{t.loc[1, 'AUC']:.4f}) "
        f"CV AUC 는 오히려 내려간다({base_cv:.4f}→{t.loc[1, 'CV AUC']:.4f}). 분할 노이즈를 성능 개선으로 읽으면 안 된다."
    )
    print(
        "  → 모집단에 따라 확실히 움직이는 것은 이탈률이다. 리포트에는 모집단과 함께 쓴다."
    )
    print("  → 답은 '빼기'도 '넣기'도 아닌 '분리해서 둘 다 보고하기'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 1. 모집단을 바꾸면 이탈률이 바뀌는 이유
#    이탈률 = 모집단 안의 이탈자 수 / 모집단의 고객 수 -> 모집단은 우리가 고르는 값
#    1회 구매 고객 1,545명(31.2%)은 이탈률 73.9%, 2회 이상 고객 3,399명(68.8%)은 이탈률 36.2%
#    전체 48.0% = 0.312 x 73.9% + 0.688 x 36.2% 로 두 집단의 가중평균일 뿐이고 어느 쪽도 대표하지 않음
#    1회 구매 고객을 빼면 이탈률이 36.2%로 내려가는 것은 고객이 충성스러워져서가 아니라
#    이탈률이 가장 높은 집단을 모집단에서 뺐기 때문
#    신규 고객(105명)을 빼면 이탈률이 거의 그대로(48.0% -> 48.2%)인 것도 빠진 사람이 적기 때문
#    생존편향: 한 번이라도 다시 온 고객(살아남은 고객)만 보고 '우리 이탈률은 36%'라고 하면 실제보다 낙관적임
#    그래서 이탈률은 반드시 모집단 정의와 함께 쓰고 집단별로 나눠서 보고함
#
# 2. 단일 분할 AUC와 CV AUC가 반대로 움직인 이유
#    단일 분할: 7:3으로 한 번 나눠 시험셋 한 곳에서만 AUC를 구함 -> 어느 고객이 시험셋에 들어갔는지에 따라 흔들림
#    CV: 서로 다른 시험셋 5곳의 AUC를 평균냄 -> 표준편차로 흔들림의 크기까지 확인할 수 있음
#    전체: 단일 0.7818 / CV 0.8044 (5개 값 0.7959 ~ 0.8167) -> 단일 분할이 5개 값보다 모두 낮은, 운 나쁜 분할이었음
#    1회 구매 제외: 단일 0.8078 / CV 0.7835 (표준편차 0.0182) -> 단일 분할이 평균보다 높은, 운 좋은 분할이었음
#    단일 분할끼리 비교하면 운 나쁜 분할(0.7818)과 운 좋은 분할(0.8078)을 비교하는 셈이어서 0.026 상승은 개선이 아님
#    모집단이 작을수록 시험셋 고객이 줄어(약 989명 -> 680명) AUC 추정이 불안정해져 표준편차가 커짐 (0.0080 -> 0.0182)
#    CV로 보면 평균은 오히려 내려갔고(-0.021) 그 크기도 표준편차와 비슷하므로 개선도 악화도 확실하지 않음
#    (1회 구매 고객은 frequency 등으로 구분이 쉬워 AUC를 올려 주었을 가능성이 있으나 이 파일에서 측정하지는 않음)
#
# 3. 결론: 분리해서 둘 다 보고하기
#    모집단에 따라 확실히 움직이는 것은 AUC가 아니라 이탈률임
#    리포트에는 이탈률을 모집단과 함께 쓰고 1회 구매와 2회 이상을 따로 보고함
#    성능 차이는 CV 평균과 표준편차로 판단하고 단일 분할의 차이만으로는 판단하지 않음
# ====================================================================================================
