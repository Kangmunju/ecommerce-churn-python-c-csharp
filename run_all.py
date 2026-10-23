"""전체 파이프라인을 한 줄로 재현한다.

python run_all.py                 # 전체 실행
python run_all.py --from 4        # 4번 단계부터 다시 실행
python run_all.py --only 7 9      # 7, 9번 단계만 실행

원본 다운로드부터 분석까지 순서대로 실행하고, 단계별 소요 시간을
reports/run_log.csv 에 남긴다. 어느 위치에서 실행해도 프로젝트 루트 기준으로 동작한다.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# 이 파일이 있는 폴더 = 프로젝트 루트
ROOT = Path(__file__).resolve().parent

# 실행 순서대로 (단계 이름, 스크립트 경로)
STEPS = [
    ("원본 다운로드", "src/download_data.py"),
    ("엑셀 → parquet", "src/build_db.py"),
    ("SQLite 적재", "src/db.py"),
    ("정제", "src/clean.py"),
    ("이탈 라벨", "src/labels.py"),
    ("피처", "src/features.py"),
    ("학습·평가", "src/train.py"),
    ("a1 EDA", "analysis/a1_eda.py"),
    ("a2 이탈 정의", "analysis/a2_label_definition.py"),
    ("a3 시점 누수", "analysis/a3_leakage.py"),
    ("a4 생존편향", "analysis/a4_survivorship.py"),
    ("a5 기대이익", "analysis/a5_profit.py"),
    ("a6 MLP 함정", "analysis/a6_mlp_pitfalls.py"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # from은 파이썬 예약어라 사용할 수 없어서 저장 이름을 start로 바꾸어 사용함
    ap.add_argument(
        "--from",
        dest="start",
        type=int,
        default=1,
        help="이 번호의 단계부터 실행 (1부터)",
    )
    ap.add_argument("--only", type=int, nargs="+", help="지정한 번호의 단계만 실행")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    for d in ("data/raw", "data/processed", "reports/figures"):
        Path(d).mkdir(parents=True, exist_ok=True)

    # Windows 콘솔(cp949)에서도 한글 출력이 깨지거나 죽지 않도록 UTF-8로 고정함
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # 실행할 단계를 고름
    # 번호, 이름, 스크립트
    selected = []
    for i, (name, script) in enumerate(STEPS, 1):
        if args.only:
            run_this = i in args.only
        else:
            run_this = i >= args.start
        if run_this:
            selected.append((i, name, script))

    # 실행할 스크립트가 하나라도 없으면 시작하지 않도록 작성함
    missing = []
    for _, _, script in selected:
        if not (ROOT / script).exists():
            missing.append(script)

    if missing:
        print("[중단] 아래 스크립트가 없습니다:\n " + "\n ".join(missing))
        return 2

    log = []
    t_all = time.time()
    for i, name, script in selected:
        print(
            f"\n{'=' * 74}\n[{i}/{len(STEPS)}] {name}  ({script})\n{'=' * 74}",
            flush=True,
        )
        t = time.time()

        # 같은 파이썬(sys.executable)으로 스크립트를 별도 프로세스로 실행함
        r = subprocess.run([sys.executable, script], env=env)
        dt = time.time() - t
        log.append((i, name, script, round(dt, 1), r.returncode))

        # 한 단계라도 실패하면 뒤 단계는 앞 단계의 결과에 의존하도록 멈춤
        if r.returncode != 0:
            print(f"\n[중단] {script} 가 코드 {r.returncode} 로 실패했습니다.")
            break

        print(f"\n ... {dt:.1f}초", flush=True)

    total = time.time() - t_all
    with open("reports/run_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "name", "script", "seconds", "returncode"])
        w.writerows(log)

    # 모든 단계의 반환 코드가 0이면 성공인 것으로 봄
    ok = True
    for entry in log:
        if entry[4] != 0:
            ok = False

    print(f"\n{'=' * 74}\n{'전체 완료' if ok else '실패'} — {total:.1f}초\n{'=' * 74}")

    for i, name, script, dt, rc in log:
        print(
            f"  {i:>2}. {name:20s} {dt:>7.1f}초   {script}"
            + ("" if rc == 0 else f"   (코드 {rc})")
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
