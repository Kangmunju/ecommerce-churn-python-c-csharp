"""Streamlit 앱(app/streamlit_app.py)이 읽을 작은 CSV 를 만든다.

python tools/export_for_app.py        # 프로젝트 루트에서 실행 (먼저 tools/export_for_c.py 를 실행해야 함)

입력: data/processed/export_for_c/reference_test_pred.csv  (customer_id, y, p_logit)
      data/processed/export_for_c/reference_features.csv   (customer_id, 피처 20개, churn, is_new)
출력: app/data/test_pred.csv                               (y, p_logit)
      app/data/feature_corr.csv                            (feature, corr_with_churn)

data/ 는 저장소에 올리지 않으므로, 배포된 앱이 읽을 수 있게 필요한 값만 따로 저장한다.
- 시험 고객 1,484명의 정답(y)과 예측 확률(p_logit) : 고객 번호는 앱에 필요 없어서 뺀다.
- 피처 20개와 이탈(churn)의 단독 상관계수 : 로지스틱 계수와 비교해 부호가 뒤집히는 피처를 찾는 데 쓴다.
"""

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "processed" / "export_for_c"
APP_DATA = ROOT / "app" / "data"

# 피처가 아닌 열 (고객 번호, 정답, 신규 여부)
NOT_FEATURES = ("customer_id", "churn", "is_new")


def export_predictions() -> int:
    """시험 고객의 정답과 예측 확률만 저장함. 저장한 고객 수를 돌려줌."""
    source = EXPORT_DIR / "reference_test_pred.csv"
    target = APP_DATA / "test_pred.csv"
    with open(source, encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))

    with open(target, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["y", "p_logit"])
        for row in rows:
            writer.writerow([row["y"], row["p_logit"]])
    return len(rows)


def export_feature_corr() -> int:
    """피처마다 이탈(churn)과의 피어슨 상관계수를 구해 저장함. 저장한 피처 수를 돌려줌."""
    source = EXPORT_DIR / "reference_features.csv"
    target = APP_DATA / "feature_corr.csv"
    with open(source, encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))

    churn = [float(row["churn"]) for row in rows]
    feature_names = [name for name in rows[0].keys() if name not in NOT_FEATURES]

    with open(target, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["feature", "corr_with_churn"])
        for name in feature_names:
            values = [float(row[name]) for row in rows]
            writer.writerow([name, statistics.correlation(values, churn)])
    return len(feature_names)


def main() -> int:
    needed = [EXPORT_DIR / "reference_test_pred.csv", EXPORT_DIR / "reference_features.csv"]
    for path in needed:
        if not path.exists():
            print(f"입력 파일이 없음: {path}")
            print("python tools/export_for_c.py 를 먼저 실행할 것")
            return 1

    APP_DATA.mkdir(parents=True, exist_ok=True)
    print(f"{export_predictions():,}명 저장: {APP_DATA / 'test_pred.csv'}")
    print(f"{export_feature_corr()}개 피처 저장: {APP_DATA / 'feature_corr.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
