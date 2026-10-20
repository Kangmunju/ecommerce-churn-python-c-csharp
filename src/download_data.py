"""UCI Online Retail II 원본을 내려받는다 (인증키·계정 불필요).

python src/download_data.py
"""

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# 프로젝트 루트에서 실행하는 것을 전제로 하기 때문에 상대경로로 경로를 작성함
RAW = Path("data/raw")
ZIP = RAW / "online_retail_ii.zip"
XLSX = RAW / "online_retail_II.xlsx"

# 실제로 받았을 때의 값
# 원본이 갱신되면 달라질 수 있음 -> 그 때에는 경고만
EXPECT_BYTES = 45_622_418


# 파일 지문을 계산
# 같은 파일이면 항상 같은 값이 나옴 -> 재현성 확인을 위함
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        # 1MB씩 끊어 읽어 update에 누적함
        # 함수가 빈 바이트를 돌려줄 때까지 반복함
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 이미 온전한 파일이 있으면 다시 받지 않고 건너뛰도록 작성함
def download() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and ZIP.stat().st_size == EXPECT_BYTES and zipfile.is_zipfile(ZIP):
        print(f"[skip] 이미 있음 {ZIP} ({ZIP.stat().st_size:,} bytes)")
        return ZIP

    # 네트워크 재시도 3회
    for attempt in range(1, 4):
        try:
            # stream=True: 응답 본문을 한 번에 받지 않고 나눠서 받음
            # User-Agent: 일부 서버는 스크립트 요청을 거부해서 브라우저처럼 보이게 함
            r = requests.get(
                URL, timeout=600, stream=True, headers={"User-Agent": "Mozilla/5.0"}
            )
            # 404, 500 같은 HTTP 오류를 예외로 바꿔서 재시도 대상으로 만듦
            r.raise_for_status()
            n = 0
            with open(ZIP, "wb") as f:
                # 응답을 한 번에 메모리에 올리지 않고 1MB씩 파일에 쓰도록 함
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            print(f"[get ] {URL}")
            print(f"       {n:,} bytes 수신")
            break
        except Exception as e:
            print(f"       시도 {attempt}/3 실패: {type(e).__name__}: {e}")
    else:
        raise SystemExit("다운로드에 3번 실패했습니다. 네트워크를 확인하세요.")

    # 받았다 != 제대로 받았다 -> 내용을 확인함
    # 에러 페이지(HTML)가 zip 이름으로 저장되는 사고를 막음
    if not zipfile.is_zipfile(ZIP):
        raise SystemExit(f"{ZIP} 가 zip이 아닙니다. 응답이 에러 페이지일 수 있습니다.")
    # 크기가 다르면 중단하지 않고 경고만 냄 (원본이 갱신됐을 수 있기 때문)
    if ZIP.stat().st_size != EXPECT_BYTES:
        print(
            f"       [warn] 크기가 기록({EXPECT_BYTES:,})과 다릅니다: {ZIP.stat().st_size:,}"
        )
        print(
            "              원본이 갱신됐을 수 있습니다. reports/DATA_ACCESS.md 를 갱신하세요."
        )
    return ZIP


# 이미 풀려 있으면 다시 풀지 않고 건너뜀 (download와 같은 멱등성)
def unzip() -> Path:
    if XLSX.exists():
        print(f"[skip] 이미 풀려 있음 {XLSX} ({XLSX.stat().st_size:,} bytes)")
        return XLSX
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        print(f"[zip ] 내용물 {names}")
        z.extractall(RAW)
    # 푼 뒤에 기대한 파일이 실제로 생겼는지 확인함 (zip 안의 파일명이 바뀐 경우 대비)
    if not XLSX.exists():
        raise SystemExit(f"{XLSX} 가 없습니다. zip 내용물을 확인하세요.")
    print(f"[ok  ] {XLSX} ({XLSX.stat().st_size:,} bytes)")
    return XLSX


# 다운로드 -> 압축 해제 -> 해시 출력 순서로 실행
def main() -> int:
    download()
    unzip()
    print(f"[sha ] zip  {sha256(ZIP)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
