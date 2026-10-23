"""10장 — 임계값은 통계가 아니라 돈이 정한다.

python analysis/a5_profit.py        # 먼저 python src/train.py 가 필요 (test 예측 확률)

이탈 예측의 결과물은 "이탈입니다"가 아니라 "이 고객에게 쿠폰을 쏠까요?" 다.
없는 숫자는 가정하되 코드 맨 위에 모으고, 틀렸을 때 답이 어떻게 바뀌는지(민감도)를 같이 낸다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT =Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import labels as L  # noqa: E402
from train import PRED, SEED, load_xy, make_logit, split  # noqa: E402

# 가정
# 실무에서는 재무팀, 마케팅팀에서 받는 숫자
# 공헌 이익률
MARGIN = 0.30

# 쿠폰 1장 원가 £
COUPON_COST = 5.00

# 이탈 예정자가 쿠폰을 받고 돌아올 확률
RESPONSE = 0.10

# R_RETAINED(유지 고객 1명의 예측창 매출 중앙값 £)는 가정이 아니라 데이터에서 실측할 것

# 임계값 0.05 ~ 0.95를 0.05 간격으로 시험함
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

# 가정
# test 1,484명 기준 기대 이익이 해당 값 미만이면 운영 비용을 감안해 '사실상 손익분기'로 봄
MIN_WORTH = 100.0


# fut: 예측창의 거래
# obs_customers: 관측창에 있던 고객 번호
# 둘을 isin으로 맞춰 관측창 고객 중 예측창에 구매한 사람만 남김
def measure_r_retained() -> tuple[float, float, int]:
    """유지 고객(예측창에 구매한 사람)의 예측창 매출 중앙값·평균·인원."""
    df = pd.read_parquet("data/processed/clean.parquet")
    fut = df[(df.invoice_date >= L.CUTOFF) & (df.invoice_date < L.LABEL_END)]
    obs_customers = df[df.invoice_date < L.CUTOFF].customer_id.unique()
    rev = fut[fut.customer_id.isin(obs_customers)].groupby("customer_id").revenue.sum()
    return float(rev.median()), float(rev.mean()), int(len(rev))


def values(
    r_retained: float,
    response: float = RESPONSE,
    coupon: float = COUPON_COST,
    margin: float = MARGIN,
) -> tuple[float, float]:

    # 진짜 이탈자에게 쓴 경우
    value_tp = response * r_retained * margin - coupon

    # 어차피 올 사람에게 쓴 경우
    value_fp = -coupon

    # FN·TN은 아무것도 하지 않으므로 0으로 체크함

    return value_tp, value_fp


# 임계값마다 '확률이 해당 값 이상이면 발송'으로 두고 TP와 FP를 세어 이익을 계산함
# pred & (y == 1): 발송했고 실제로 이탈한 고객
# pred & (y == 0): 발송했지만 실제로는 유지될 고객
def profit_curve(
    y: np.ndarray, prob: np.ndarray, value_tp: float, value_fp: float
) -> pd.DataFrame:
    rows = []
    for thr in THRESHOLDS:
        pred = prob >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rows.append(
            {
                "임계값": thr,
                "발송": tp + fp,
                "TP": tp,
                "FP": fp,
                # 발송이 0건인 경우 0으로 나누는 에러를 방지하기 위해 np.nan으로 비어있음을 알림
                "정밀도": tp / (tp + fp) if tp + fp else np.nan,
                "기대이익": tp * value_tp + fp * value_fp,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if not PRED.exists():
        raise SystemExit(f"{PRED} 가 없습니다. 먼저 python src/train.py 를 실행하세요.")
    t = pd.read_parquet(PRED)
    y, prob = t["y"].to_numpy(), t["p_logit"].to_numpy()

    r_med, r_mean, n_ret = measure_r_retained()
    vtp, vfp = values(r_med)
    print(
        f"[가정] 마진 {MARGIN:.0%} · 쿠폰 원가 £{COUPON_COST:.2f} · 반응률 {RESPONSE:.0%}"
    )
    print(
        f"[실측] 유지 고객 {n_ret:,}명의 예측창 매출  중앙값 £{r_med:,.2f}  (평균 £{r_mean:,.2f})"
    )
    print(
        f"       → 맞게 쏜 1건 +£{vtp:.2f} / 헛되이 쏜 1건 −£{-vfp:.2f}  (가치비 {vtp / -vfp:.1f}배)\n"
    )

    pc = profit_curve(y, prob, vtp, vfp)
    best = pc.loc[pc.기대이익.idxmax()]
    base = pc.loc[pc.임계값 == 0.50].iloc[0]
    show = pc.copy()
    show["표시"] = np.where(
        show.임계값 == best.임계값,
        "← 최대",
        np.where(show.임계값 == 0.50, "← sklearn 기본값", ""),
    )
    print(show.round({"정밀도": 2, "기대이익": 2}).to_string(index=False))
    n_test = len(y)
    gain = best.기대이익 - base.기대이익
    print(
        f"\n  기본값 0.50 → 발송 {int(base.발송):,}건, 기대이익 £{base.기대이익:,.0f}"
    )
    print(
        f"  최적  {best.임계값:.2f} → 발송 {int(best.발송):,}건, 기대이익 £{best.기대이익:,.0f}"
    )
    print(
        f"  차이 £{gain:,.0f} (test {n_test:,}명 기준; 전체 {len(load_xy()[1]):,}명으로 환산하면 약 "
        f"£{gain * len(load_xy()[1]) / n_test:,.0f})"
    )
    pc.to_csv("reports/a5_profit_curve.csv", index=False, encoding="utf-8")

    # 검증 1
    # 임계값을 test 에서 고르면 낙관적이므로 학습셋 OOF(out-of-fold) 확률에서 고르고 test 에 적용함
    X, y_all = load_xy()
    Xtr, Xte, ytr, yte = split(X, y_all)
    oof = cross_val_predict(
        make_logit(),
        Xtr,
        ytr,
        method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    )[:, 1]
    thr_oof = float(
        profit_curve(ytr, oof, vtp, vfp).sort_values("기대이익").iloc[-1].임계값
    )
    applied = profit_curve(y, prob, vtp, vfp).set_index("임계값").loc[thr_oof]
    print(
        f"\n[검증] 임계값을 학습셋 out-of-fold 확률에서 고르면 {thr_oof:.2f} → "
        f"test 적용 시 발송 {int(applied.발송):,}건, 기대이익 £{applied.기대이익:,.0f} "
        f"(test 에서 고른 최적 £{best.기대이익:,.0f}, 기본값 0.50 £{base.기대이익:,.0f})"
    )

    # 검증 2
    # 확률이 진짜 확률인지 확인
    # 기대이익 계산은 이게 맞을 때만 의미가 있음
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=5, strategy="quantile")
    cal = pd.DataFrame({"예측 확률 평균": mean_pred, "실제 이탈 비율": frac_pos}).round(
        3
    )
    print("\n[보정 확인] 예측 확률 5분위별 — 두 열이 비슷해야 확률로 쓸 수 있다")
    print(cal.to_string(index=False))
    cal.to_csv("reports/a5_calibration.csv", index=False, encoding="utf-8")

    # 민감도
    # 가정이 틀린 경우
    # 반응률 4가지 * 쿠폰 원가 3가지 = 12가지 조합
    # 각 조합마다 최적 임계값과 기대이익을 다시 구하고 그 가정이라면 실행할 만한가를 판정함
    rows = []
    for resp in (0.03, 0.05, 0.10, 0.20):
        for cc in (2.0, 5.0, 15.0):
            a, b = values(r_med, response=resp, coupon=cc)
            c = profit_curve(y, prob, a, b)
            bb = c.loc[c.기대이익.idxmax()]

            # 최대 기대이익을 기준으로 권고를 정함
            if bb.기대이익 <= 0:
                advice = "하지 말 것"
            elif bb.기대이익 >= MIN_WORTH:
                advice = "실행"
            else:
                advice = "사실상 손익분기"

            rows.append(
                {
                    "반응률": resp,
                    "쿠폰원가": cc,
                    "최적 임계값": bb.임계값,
                    "발송": int(bb.발송),
                    "기대이익": round(bb.기대이익),
                    "권고": advice,
                }
            )

    sens = pd.DataFrame(rows)
    print("\n[민감도] 가정이 틀렸다면")
    print(sens.to_string(index=False))
    sens.to_csv("reports/a5_sensitivity.csv", index=False, encoding="utf-8")

    with open("reports/a5_assumptions.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "MARGIN": MARGIN,
                "COUPON_COST": COUPON_COST,
                "RESPONSE": RESPONSE,
                "MIN_WORTH_test_gbp": MIN_WORTH,
                "R_RETAINED_median_measured": round(r_med, 2),
                "R_RETAINED_mean_measured": round(r_mean, 2),
                "VALUE_TP": round(vtp, 2),
                "VALUE_FP": round(vfp, 2),
                "note": "R_RETAINED 만 실측이고 나머지는 가정. 증분 효과는 관찰 데이터로 알 수 없다.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ====================================================================================================
# [설계 메모]
# ====================================================================================================
# 1. 기대이익 = TP 수 x 14.05 + FP 수 x (-5) 인 이유
#    이익은 '아무것도 안 했을 때(0원)'와 비교한 추가 이익임
#    TP(진짜 이탈자에게 발송): 0.10(반응률) x 635.01(유지 고객 매출 중앙값) x 0.30(마진) - 5(쿠폰 원가) = +14.05
#      이탈 고객 100명에게 보내면 10명이 돌아와 마진 190.5 x 10 = 1,905, 쿠폰 100장 = 500 -> 순이익 1,405 (한 명당 14.05)
#    FP(어차피 올 사람에게 발송): 쿠폰이 만든 추가 매출이 없고 쿠폰 원가만 나감 = -5
#    FN, TN: 발송하지 않았으므로 0 (놓친 이탈자는 '못 번 돈'이지 '나간 돈'이 아님)
#    고객마다 이익이 정해져 있고 서로 영향을 주지 않으므로 칸별 고객 수 x 한 명당 이익을 더하면 전체 이익이 됨
#    확인: 0.50은 531 x 14.05 - 245 x 5 = 약 6,236 / 0.30은 643 x 14.05 - 413 x 5 = 약 6,969
#    0.50 -> 0.30으로 낮추면 TP +112명(+1,573.6), FP +168명(-840) = +733.6이 되어 표의 차이 734와 같음
#    이익 분기점: 확률 p인 고객에게 보낼 때의 기대이익 p x 14.05 - (1-p) x 5 가 0 이상이 되려면 p >= 5 / 19.05 = 약 0.26
#    확률이 정확하다면 이탈 확률 26% 이상이면 보내는 쪽이 이득이므로 기본값 0.50은 26 ~ 50% 구간을 놓침
#    가치비(14.05 : 5 = 2.8 : 1)가 클수록 분기점이 낮아짐
#
# 2. 임계값을 OOF(out-of-fold) 확률에서 고르는 이유
#    test에서 임계값을 고르면 정답을 고른 데이터로 성능을 평가하는 것이라 낙관적임
#    학습셋 고객을 최종 모델로 다시 예측하면 이미 본 문제를 푸는 것이라 확률이 지나치게 자신 있게 나옴
#    cross_val_predict는 학습셋을 5조각으로 나눠 각 조각을 그 조각을 빼고 학습한 모델로 예측함
#    -> 모든 학습셋 고객이 자기를 보지 않은 모델의 확률을 갖게 되어 처음 보는 고객에 대한 예측과 같은 성격이 됨
#    결과: OOF에서 고른 0.25를 test에 적용하면 6,909 (test에서 고른 최적 6,969, 기본값 0.50 6,236)
#    기본값 대비 이익 673은 test에서 고른 경우 734의 약 92% -> 0.5보다 낮은 임계값이 좋다는 결론은 test에 맞춘 것이 아님
#    이번에는 차이가 작지만 이익 곡선이 0.25 ~ 0.30에서 평평하기 때문이고 원리는 항상 지켜야 함
#
# 3. 확률이 진짜 확률이어야 하는 이유 (보정)
#    기대이익은 모델이 낸 확률의 크기가 맞을 때만 의미가 있음 (순서만 맞는 AUC와 다름)
#    5분위별 예측 확률 평균과 실제 이탈 비율이 대체로 비슷함 (0.313 vs 0.296, 0.511 vs 0.527, 0.633 vs 0.633)
#    가장 낮은 구간은 예측 0.086, 실제 0.135로 이탈 확률이 낮은 고객의 이탈을 다소 과소평가함
#
# 4. 가정과 한계
#    마진 30%, 쿠폰 원가 5, 반응률 10%는 데이터에 없는 가정이고 R_RETAINED만 데이터에서 실측함
#    R_RETAINED는 평균(1,758.29)이 아니라 중앙값(635.01)을 씀 (소수의 큰 고객이 평균을 끌어올림, a1_eda.py 참고)
#    민감도: 반응률 3% + 쿠폰 원가 5는 사실상 손익분기(기대이익 10), 반응률 3% + 쿠폰 원가 15는 손실(-210)이라 하지 말 것
#    결론은 '임계값 0.30이 최적'이 아니라 '반응률이 충분하고 쿠폰이 싸면 0.5보다 낮은 임계값이 이익, 아니면 하지 않는 것이 낫다'임
#    쿠폰이 실제로 고객을 돌아오게 하는지(증분 효과)는 관찰 데이터로 알 수 없고 A/B 실험이 필요함
#    기대이익은 가정 위의 기댓값이지 실제 이익이 아님
# ====================================================================================================
