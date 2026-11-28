"""Streamlit 앱(app/streamlit_app.py)이 읽을 작은 CSV 를 만든다.

python tools/export_for_app.py        # 프로젝트 루트에서 실행 (먼저 tools/export_for_c.py 를 실행해야 함)

입력: data/processed/export_for_c/reference_test_pred.csv  (customer_id, y, p_logit)
출력: app/data/test_pred.csv                               (y, p_logit)

data/ 는 저장소에 올리지 않으므로, 배포된 앱이 읽을 수 있게 시험 고객 1,484명의
정답(y)과 예측 확률(p_logit)만 따로 저장한다. 고객 번호는 앱에 필요 없어서 뺀다.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "export_for_c" / "reference_test_pred.csv"
TARGET = ROOT / "app" / "data" / "test_pred.csv"


def main() -> int:
    if not SOURCE.exists():
        print(f"입력 파일이 없음: {SOURCE}")
        print("python tools/export_for_c.py 를 먼저 실행할 것")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCE, encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))

    with open(TARGET, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["y", "p_logit"])
        for row in rows:
            writer.writerow([row["y"], row["p_logit"]])

    print(f"{len(rows):,}명 저장: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
