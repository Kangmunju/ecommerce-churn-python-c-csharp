"""11장 — MLP 를 표(정형) 데이터에 올릴 때.

python analysis/a6_mlp_pitfalls.py

  1. 표준화를 빼면 얼마나 망가지나   (3권 경사하강)
  2. 시드를 10번 바꾸면 얼마나 흔들리나 (결론의 신뢰도)
  3. 층을 키우면 좋아지나             (4권 깊은 신경망의 현실)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import SEED, load_xy, make_logit, make_mlp, split  # noqa: E402

# 수렴 경고는 숨김
# 아래 실험에서 원인을 직접 확인할 것
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def auc(model, Xtr, Xte, ytr, yte) -> float:
    return roc_auc_score(yte, model.fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def raw_mlp(seed=SEED) -> MLPClassifier:
    """표준화 없이 원본 그대로 넣는 MLP (make_mlp 와 하이퍼파라미터는 동일)."""
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )


def auc_std_error(a: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil(1982) 의 AUC 표준오차 근사. test 표본이 작아서 생기는 불확실성의 크기."""

    # AUC: (이탈자, 유지 고객) 쌍을 비교해 이탈자가 더 높게 평가된 비율
    # 같은 사람이 여러 쌍에 겹쳐 쓰이므로 쌍들이 서로 독립이 아니고 그만큼 분산이 커짐
    # q1: 이탈자 2명이 같은 유지 고객 1명보다 모두 높게 평가될 확률
    #     같은 유지 고객을 공유하는 두 비교가 함께 이길 확률
    q1 = a / (2 - a)

    # q2: 이탈자 1명이 유지 고객 2명보다 모두 높게 평가될 확률
    #     같은 이탈자를 공유하는 두 비교가 함께 이길 확률
    q2 = 2 * a * a / (1 + a)

    # 분산 = 쌍 하나의 변동 a(1-a) + 같은 유지 고객을 공유하는 쌍들의 공분산 (q1 - a^2)
    #       + 같은 이탈자를 공유하는 쌍들의 공분산 (q2 - a^2)
    # 앞의 (n_pos - 1), (n_neg - 1)은 같은 사람을 공유하는 다른 쌍의 개수를 나타냄
    # 전체를 n_pos x n_neg 로 나누어 모든 쌍을 평균낸 값(AUC)의 분산을 계산함
    # q1, q2 식은 점수가 지수분포를 따른다고 가정한 Hanley & McNeil 의 근사식
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (
        n_pos * n_neg
    )

    # 분산의 제곱근이 표준오차
    return float(np.sqrt(var))


def n_params(hidden, n_in) -> int:

    # 입력층, 은닉층들, 출력층(1개)의 뉴런 수를 순서대로 나열함
    sizes = [n_in, *hidden, 1]
    total = 0

    # 이웃한 두 층마다 가중치(앞 층 x 뒤 층) + 편향(뒤 층)을 더함
    # 앞 층의 모든 뉴런이 뒤 층의 모든 뉴런과 연결되므로 가중치는 앞 층 x 뒤 층 개
    # 뒤 층의 뉴런마다 편향이 하나씩 있으므로 편향은 뒤 층 개
    # 예: 입력 20, 은닉 (32, 16), 출력 1 -> (20x32+32) + (32x16+16) + (16x1+1) = 1,217
    for i in range(len(sizes) - 1):
        total += sizes[i] * sizes[i + 1] + sizes[i + 1]
    return total


# 피처마다 크기가 크게 차이남
# 모든 피처를 비슷한 범위로 맞추기 위해 표준화를 진행함
# 4가지 조합(MLP / 로지스틱 * 표준화 있음/ 없음 )모두 돌려서 비교
def main() -> int:
    X, y = load_xy()
    Xtr, Xte, ytr, yte = split(X, y)
    Path("reports").mkdir(exist_ok=True)

    # 1. 표준화
    print(
        f"[스케일] 피처별 크기 차이 — monetary 최대 {X.monetary.max():,.0f} vs return_rate 최대 {X.return_rate.max():.2f}\n"
    )
    a_mlp_sc = auc(make_mlp(), Xtr, Xte, ytr, yte)
    a_mlp_raw = auc(raw_mlp(), Xtr, Xte, ytr, yte)
    a_lr_sc = auc(make_logit(), Xtr, Xte, ytr, yte)
    a_lr_raw = auc(
        LogisticRegression(max_iter=2000, random_state=SEED), Xtr, Xte, ytr, yte
    )
    sc = pd.DataFrame(
        [
            {"모델": "MLP", "표준화": "있음", "AUC": a_mlp_sc},
            {"모델": "MLP", "표준화": "없음", "AUC": a_mlp_raw},
            {"모델": "로지스틱", "표준화": "있음", "AUC": a_lr_sc},
            {"모델": "로지스틱", "표준화": "없음", "AUC": a_lr_raw},
        ]
    )
    print(sc.round(4).to_string(index=False))
    print(
        f"\n  → 표준화를 빼면 MLP {a_mlp_raw - a_mlp_sc:+.4f}, 로지스틱 {a_lr_raw - a_lr_sc:+.4f}"
    )
    sc.to_csv("reports/a6_scaling.csv", index=False, encoding="utf-8")

    # 2. 시드 10번
    aucs = []
    for i in range(10):
        aucs.append(auc(make_mlp(seed=i), Xtr, Xte, ytr, yte))
    s = np.array(aucs)
    wins = int((s > a_lr_sc).sum())
    diff = s.mean() - a_lr_sc
    print(
        f"\n[시드 10개] MLP AUC 평균 {s.mean():.4f} ± {s.std():.4f}   최소 {s.min():.4f} / 최대 {s.max():.4f}"
    )
    print(f"  값들: {np.round(s, 4)}")
    print(f"  로지스틱 AUC {a_lr_sc:.4f} (시드에 거의 무관)")
    print(f"  MLP 가 로지스틱을 이긴 시드: {wins}/10")
    # 평균 차이가 시드 간 표준편차(흔들림)보다 작은지로 판정함
    if abs(diff) < s.std():
        verdict = "흔들림보다 작다 → 'MLP 가 이겼다'고 쓸 근거가 없다"
    else:
        verdict = "흔들림보다 크다 → 차이가 있다고 볼 수 있다"
    print(f"  평균 차이 {diff:+.4f} vs 시드 간 표준편차 {s.std():.4f}: {verdict}")
    pd.DataFrame({"seed": range(10), "mlp_auc": s, "logit_auc": a_lr_sc}).to_csv(
        "reports/a6_seeds.csv", index=False, encoding="utf-8"
    )

    # 3. 층 크기
    rows = []
    for h in [(8,), (16,), (32, 16), (64, 32), (128, 64, 32), (256, 128, 64, 32)]:
        a = []
        for i in range(3):
            a.append(auc(make_mlp(seed=i, hidden=h), Xtr, Xte, ytr, yte))
        rows.append(
            {
                "은닉층": str(h),
                "뉴런 수": sum(h),
                "파라미터": n_params(h, X.shape[1]),
                "AUC 평균(시드 3개)": np.mean(a),
                "표준편차": np.std(a),
            }
        )
    cap = pd.DataFrame(rows)
    print("\n[층 키우기]")
    print(cap.round(4).to_string(index=False))
    print(f"\n  로지스틱 {a_lr_sc:.4f} — 파라미터 {X.shape[1] + 1}개")
    first, last, top = cap.iloc[0], cap.iloc[-1], cap["AUC 평균(시드 3개)"].max()
    lo, sd = cap["AUC 평균(시드 3개)"].min(), cap["표준편차"].max()
    print(
        f"  → 뉴런을 {int(first['뉴런 수'])}개에서 {int(last['뉴런 수'])}개로 {last['뉴런 수'] / first['뉴런 수']:.0f}배 늘려도 "
        f"AUC 가 {first['AUC 평균(시드 3개)']:.4f} → {last['AUC 평균(시드 3개)']:.4f} (구성별 범위 {lo:.4f}~{top:.4f}, "
        f"시드 간 표준편차 최대 {sd:.4f})."
    )
    se = auc_std_error(a_lr_sc, int(yte.sum()), int(len(yte) - yte.sum()))
    print(
        f"    test {len(yte):,}명으로 잰 AUC 의 표집 오차(표준오차)는 약 {se:.4f} 다. "
        f"최고 구성과 로지스틱의 차이({top - a_lr_sc:+.4f})는 그보다 작고, 구성 간 순서도 단조롭지 않다"
        f"(뉴런이 늘수록 좋아지지 않는다) → 용량을 키워서 얻는 이득을 말할 수 없다."
    )
    cap.to_csv("reports/a6_capacity.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 1. 표준화를 빼면 MLP가 망가지는 이유
#    MLP는 경사하강법으로 학습하고 가중치의 기울기는 입력값에 비례함
#    monetary(최대 413,459)와 return_rate(최대 1.00)가 섞이면 monetary 쪽 기울기가 수십만 배 커짐
#    같은 학습률(1e-3)로 monetary 가중치는 한 번에 너무 크게 움직이고 return_rate 가중치는 거의 움직이지 않음
#    첫 층의 계산이 monetary 하나에 지배되어 다른 피처는 사실상 반영되지 않음
#    결과: MLP AUC 0.7886 -> 0.7189 (-0.0697) / 로지스틱 0.7818 -> 0.7842 (+0.0024)로 로지스틱은 거의 그대로
#    로지스틱이 덜 영향받는 이유는 이 파일에서 측정하지 않았음 (관찰된 결과)
#    로지스틱도 표준화를 하는 이유는 계수의 크기를 서로 비교하기 위해서임 (train.py 계수 표 참고)
#
# 2. 시드 간 표준편차와 표준오차는 서로 다른 흔들림
#    시드 간 표준편차(0.0035): 같은 시험셋에서 가중치의 초기값만 바꿨을 때 AUC가 흔들리는 크기 (모델 쪽 흔들림)
#    표준오차(0.0120): 시험셋 고객이 달라지면(1,484명을 다시 뽑으면) AUC가 흔들리는 크기 (데이터 쪽 흔들림)
#    두 흔들림이 모두 있으므로 모델 간 차이가 둘 다보다 커야 진짜 차이라고 말할 수 있음
#    MLP와 로지스틱의 평균 차이 +0.0018은 0.0035보다도 작고 0.0120보다 훨씬 작음
#    시드를 평균내어 모델 쪽 흔들림을 줄여도 데이터 쪽 흔들림은 그대로 남음
#
# 3. 단일 시드의 MLP 0.7886을 대표값으로 쓰지 않는 이유
#    train.py의 0.7886은 시드 42 한 번의 결과이고 시드 10개 평균(0.7836 ± 0.0035)보다 높은 쪽
#    그래서 MLP의 성능은 시드 평균과 표준편차를 함께 적음
#
# 4. 층을 키워도 좋아지지 않음
#    뉴런 8 -> 480개(파라미터 177 -> 48,641)로 늘려도 AUC는 0.7804 -> 0.7866, 구성 간 범위 0.7804 ~ 0.7903
#    뉴런이 늘수록 좋아지는 경향이 없고 최고 구성과 로지스틱의 차이(+0.0085)는 표준오차(0.0120)보다 작음
#    로지스틱은 파라미터 21개(피처 20개 + 절편 1개)로 같은 수준의 AUC를 냄
#    표 데이터에서는 모델의 크기가 아니라 피처(features.py)가 성능을 만듦
# ====================================================================================================
