"""탐색 — 그림 4장 (reports/figures/a1_eda.png).

python analysis/a1_eda.py        # 프로젝트 루트에서 실행
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# analysis/ 의 한 단계 위(프로젝트 루트)에 있는 src/ 폴더를 모듈 검색 경로에 추가함
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import labels as L  # noqa: E402
from plotstyle import GREEN, RED, apply  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CLEAN =Path("data/processed/clean.parquet")
FEAT = Path("data/processed/features.parquet")
OUT = Path("reports/figures/a1_eda.png")


def main() -> int:
    for p in (CLEAN, FEAT):
        if not p.exists():
            raise SystemExit(f"{p} 가 없습니다. 앞 단계를 먼저 실행하세요.")
    apply()
    df = pd.read_parquet(CLEAN)
    f = pd.read_parquet(FEAT)

    # 월별 매출 합계(MS = 매월 1일 기준) / 단위를 천 파운드로 줄임
    monthly = df.set_index("invoice_date").revenue.resample("MS").sum() / 1000
    last_month_days = df.invoice_date.max().day
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (1) 월별 매출
    # 마지막 달은 9일 뿐이라 '급감'처럼 보이는 것에 주의
    a = ax[0, 0]
    a.plot(monthly.index, monthly.values, color=GREEN, marker="o", ms=3, lw=1.5)
    a.axvline(L.CUTOFF, color="#e65100", ls="--", lw=1.2)
    a.set_title("월별 매출 (£k) · 점선=기준일")
    a.annotate(
        f"마지막 달은 {last_month_days}일치",
        (monthly.index[-1], monthly.values[-1]),
        xytext=(0.55, 0.06),
        textcoords="axes fraction",
        fontsize=8,
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED),
    )

    # (2) 구매 빈도 — 30회가 넘는 고객은 30으로 잘라서 보여 줌
    a = ax[0, 1]
    a.hist(f.frequency.clip(upper=30), bins=30, color=GREEN)
    a.set_title("구매 빈도 분포 (30회에서 자름)")
    a.set_xlabel("주문 횟수")

    # (3) recency 분포 — 유지 vs 이탈
    a = ax[1, 0]
    bins = np.linspace(0, 560, 57)
    a.hist(f[f.churn == 0].recency, bins=bins, alpha=0.6, color=GREEN, label="유지")
    a.hist(f[f.churn == 1].recency, bins=bins, alpha=0.6, color=RED, label="이탈")
    a.set_title("recency 분포 (일)")
    a.legend()

    # (4) 누적 매출 집중도 — 매출이 큰 고객부터 쌓아 올린 비율
    a = ax[1, 1]
    m = f.monetary.sort_values(ascending=False).to_numpy()
    cum = np.cumsum(m) / m.sum()
    a.plot(np.arange(1, len(m) + 1), cum, color=GREEN)
    a.set_title("누적 매출 집중도")
    a.set_xlabel("고객 수 (매출 순)")

    # tight_layout()으로 칸 사이의 제목과 글자가 겹치지 않게 간격을 조정함
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)

    top1 = max(1, int(round(len(m) * 0.01)))
    print(f"[a1] 그림 저장 {OUT}")
    print(
        f"     상위 1%({top1}명)가 매출의 {cum[top1 - 1] * 100:.1f}% — 평균 대신 중앙값을 쓰는 이유 (10장)"
    )
    print(
        f"     monetary 최대 £{m[0]:,.0f} = 평균 £{m.mean():,.0f} 의 {m[0] / m.mean():.0f}배 (MLP 표준화 필요, 11장)"
    )
    print(
        f"     마지막 달({monthly.index[-1].strftime('%Y-%m')})은 {last_month_days}일치뿐 — 월별 비교 시 주의"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
