"""matplotlib 한글 폰트·공통 스타일. 그래프의 한글이 □ 로 깨지는 문제를 막는다.

from plotstyle import apply; apply()
"""

import warnings

import matplotlib

# 화면 없이 파일로만 저장 (서버·CI 에서도 동작)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 한글을 지원하는 글꼴 후보 (앞에 있는 것부터 찾음)
CANDIDATES = [
    "Malgun Gothic",
    "AppleGothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]
# 그래프 공통 색
# 초록 = 유지
# 빨강 = 이탈
# 회색 = 보조
GREEN, RED, GRAY = "#2b6e3e", "#c0392b", "#7f8c8d"


def apply() -> None:

    # 컴퓨터에 설치된 글꼴 이름을 모음
    installed = set()
    for f in font_manager.fontManager.ttflist:
        installed.add(f.name)

    # 후보 중 처음으로 설치된 글꼴을 기본 글꼴로 지정하고 멈춤
    for name in CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        warnings.warn(
            "한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다 "
            f"(후보: {', '.join(CANDIDATES)})"
        )
    # 한글 글꼴에서 마이너스(-) 기호가 네모로 깨지는 것을 막음
    plt.rcParams["axes.unicode_minus"] = False

    # 위쪽과 오른쪽 테두리를 없애 깔끔하게 보이게 함
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # 연한 격자선을 켬
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["figure.dpi"] = 110
