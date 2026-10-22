"""학습과 평가 — 베이스라인을 먼저 세우고 올라간다.

python src/train.py

0. 다수 클래스(유지)로 전부 찍기 → 1. 열 하나 규칙 → 2. 로지스틱 → 3. MLP
"""

import sys
from pathlib import Path

import numpy as np
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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 재현성을 위해 모든 난수에 같은 시드를 사용함
SEED = 42
FEAT = Path("data/processed/features.parquet")
PRED = Path("data/processed/test_pred.parquet")

# 입력(X)에서 빼는 열
# churn : 정답
# is_new : 신규 고객 구분용 (입력에서 제외)
FEATURES_NOT_X = ["churn", "is_new"]


# 입력 X(피처 20개)와 정답 y(0/1 배열)을 돌려줌
def load_xy():
    f = pd.read_parquet(FEAT)
    return f.drop(columns=FEATURES_NOT_X), f["churn"].to_numpy()


# 분할이 파일마다 다르면 결과를 서로 비교할 수 없어 분할을 함수 하나에 모아 두어야 한다고 판단함
# 학습에 한 번도 쓰지 않은 데이터로 시험셋을 구성함
def split(X, y):
    """모든 분석(analysis/)이 같은 분할을 쓰도록 한 곳에서 정의한다."""
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


def make_logit() -> Pipeline:

    # 테스트셋 통계가 학습에 새어 들지 않도록 표준화를 Pipeline 안에서 수행함
    return Pipeline(
        [
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )


# **kw: make_mlp에 없는 옵션을 MLPClassifier로 그대로 넘기는 통로 (이미 적어 둔 옵션은 중복해서 넘길 수 없음)
def make_mlp(seed: int = SEED, hidden=(32, 16), **kw) -> Pipeline:
    return Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=600,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=seed,
                    **kw,
                ),
            ),
        ]
    )


def report(name, y_true, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "model": name,
        "AUC": roc_auc_score(
            y_true, prob
        ),  # 이탈할 사람에게 유지할 사람보다 높은 확률을 주는가
        "PR-AUC": average_precision_score(
            y_true, prob
        ),  # 이탈이라고 한 것 중 실제 이탈 비율과 포착률의 균형
        "accuracy": accuracy_score(y_true, pred),  # 전체 중 맞힌 비율
        "precision": precision_score(
            y_true, pred, zero_division=0
        ),  # 이탈이라고 한 것 중 진짜 이탈 비율
        "recall": recall_score(
            y_true, pred, zero_division=0
        ),  # 진짜 이탈자 중 잡아낸 비율
        "F1": f1_score(y_true, pred, zero_division=0),
        "Brier": brier_score_loss(y_true, prob),  # 확률이 얼마나 정확한가
    }


# 마지막 구매 후 N일 이상이면 이탈로 본다는 가장 단순한 규칙을 잡고 진행함
# N을 10일부터 395일까지 5일 간격으로 시험하며 F1이 가장 높은 N을 고름 -> 결과 80일
# 시험셋을 보고 N을 고르면 성능이 부풀려질 가능성이 있어 기준일을 학습셋에서만 고르는 것으로 구현함
def best_rule_threshold(Xtr: pd.DataFrame, ytr: np.ndarray):
    """'recency 가 N일 이상이면 이탈' - 기준일은 학습셋에서만 고름."""
    best_thr, best_f1 = None, -1.0
    for t in range(10, 400, 5):
        s = f1_score(ytr, (Xtr["recency"] >= t).astype(int))
        if s > best_f1:
            best_thr, best_f1 = t, s
    return best_thr, best_f1


def main() -> int:
    if not FEAT.exists():
        raise SystemExit(
            f"{FEAT} 가 없습니다. 먼저 python src/features.py 를 실행하세요."
        )
    X, y = load_xy()
    Xtr, Xte, ytr, yte = split(X, y)
    print(
        f"[분할] train {len(Xtr):,}명 (이탈 {ytr.mean() * 100:.1f}%) · "
        f"test {len(Xte):,}명 (이탈 {yte.mean() * 100:.1f}%)"
    )

    thr, f1tr = best_rule_threshold(Xtr, ytr)
    print(f"[규칙] 학습셋 최적 기준 recency >= {thr}일 (train F1={f1tr:.4f})")

    logit = make_logit().fit(Xtr, ytr)
    mlp = make_mlp().fit(Xtr, ytr)

    # 시험셋 고객마다 이탈할 확률(두 번째 열)을 리턴
    p_logit = logit.predict_proba(Xte)[:, 1]
    p_mlp = mlp.predict_proba(Xte)[:, 1]

    rows = [
        # 다수 클래스는 유지(52.0%)
        # 전부 '유지' 상태로 두고 아무 것도 하지 않은 기준선을 설정함
        report("0. 전부 '유지'로 찍기(다수 클래스)", yte, np.zeros(len(yte))),
        report(
            f"1. 규칙: recency >= {thr}일",
            yte,
            (Xte["recency"] >= thr).astype(float).to_numpy(),
        ),
        report("2. 로지스틱 회귀", yte, p_logit),
        report("3. MLP (32, 16)", yte, p_mlp),
    ]

    scores = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 160)
    print("\n" + scores.round(4).to_string())
    all_churn = report("(참고) 전부 '이탈'로 표기", yte, np.ones(len(yte)))
    print(
        f"\n※ 참고: 전부 '이탈'로 표기하면 정확도 {all_churn['accuracy']:.4f}, "
        f"정밀도 {all_churn['precision']:.4f}, 재현율 {all_churn['recall']:.1f}"
    )

    # 전체 데이터를 5조각으로 나눠서 한 조각씩 시험용으로 쓰고 나머지 네 조각으로 학습
    # 시험셋 한 번의 결과는 신뢰하기 어려우므로 5번 반복해 안정성을 확보함
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print("\n[5-fold CV AUC]")
    cvrows = []
    for name, m in [("로지스틱", make_logit()), ("MLP", make_mlp())]:
        s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        print(f"  {name:8s} {s.mean():.4f} ± {s.std():.4f}   {np.round(s, 4)}")
        row = {"model": name, "mean": s.mean(), "std": s.std()}
        for i in range(len(s)):
            row[f"fold{i + 1}"] = s[i]
        cvrows.append(row)

    cm = confusion_matrix(yte, (p_logit >= 0.5).astype(int))
    print("\n[혼동행렬] 로지스틱, 임계값 0.5")
    print(
        f"          예측 유지  예측 이탈\n  실제 유지   {cm[0, 0]:>5}   {cm[0, 1]:>5}\n  실제 이탈   {cm[1, 0]:>5}   {cm[1, 1]:>5}"
    )
    print(
        f"  이탈자 {cm[1].sum()}명 중 {cm[1, 1]}명 포착 (재현율 {cm[1, 1] / cm[1].sum() * 100:.1f}%), "
        f"유지될 고객 {cm[0, 1]}명에게도 '이탈' 판정"
    )

    coef = pd.Series(logit.named_steps["clf"].coef_[0], index=X.columns)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)
    print("\n[계수] 표준화 기준, 절대값 상위 10개")
    print(top.head(10).round(3).to_string())

    Path("reports").mkdir(exist_ok=True)
    scores.to_csv("reports/model_scores.csv", encoding="utf-8")
    pd.DataFrame(cvrows).to_csv("reports/cv_scores.csv", index=False, encoding="utf-8")
    top.rename("coef").to_csv("reports/logit_coefficients.csv", encoding="utf-8")
    pd.DataFrame(
        {"customer_id": Xte.index, "y": yte, "p_logit": p_logit, "p_mlp": p_mlp}
    ).to_parquet(PRED, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 1. 결과 읽는 법
#    AUC는 전부 '유지' 0.50 -> 규칙 0.68 -> 로지스틱 0.78로 오르지만 로지스틱과 MLP는 거의 같음
#    시험셋 한 번: 0.7818(로지스틱) vs 0.7886(MLP) / 5-fold CV: 0.8044 vs 0.8046
#    복잡한 모델이 아니라 피처(features.py)가 성능을 만듦
#    시험셋 한 번의 AUC는 우연에 흔들릴 수 있으므로 CV로 안정성을 함께 확인함
#    (시험셋 한 번의 0.78은 CV 5개 값 0.795 ~ 0.817보다 낮음)
#
# 2. 지표 7개와 정확도의 함정
#    AUC: 이탈할 사람에게 유지할 사람보다 높은 확률을 주는가 (0.5 = 동전 던지기, 1.0 = 완벽)
#    PR-AUC: 이탈이라고 한 것 중 실제 이탈 비율과 포착률의 균형 (기준선은 이탈 비율 0.48)
#    accuracy / precision / recall / F1: 확률을 임계값 0.5로 0과 1로 바꿔서 계산함
#    Brier: 확률이 얼마나 정확한가 (작을수록 좋음)
#    전부 '유지'로 찍어도 정확도는 52.0%, 전부 '이탈'로 찍으면 재현율은 100%
#    정확도 하나만 보면 속으므로 AUC와 precision, recall을 함께 봄
#
# 3. 평가에서 누수를 막는 세 가지
#    표준화는 Pipeline 안에서 수행함 -> 시험셋의 평균과 표준편차가 학습에 새어 들지 않음
#    규칙의 기준일(recency N일)은 학습셋에서만 고름 -> 시험셋에 맞춘 규칙이 되지 않음
#    분할은 split() 한 곳에서 정의함 -> analysis/의 모든 파일이 같은 학습/시험 분할을 씀
# ====================================================================================================
