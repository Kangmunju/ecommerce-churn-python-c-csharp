# 데이터 접근 기록

| 항목 | 내용 |
|---|---|
| 데이터셋 | Online Retail II (UCI Machine Learning Repository #502) |
| 만든 사람 | Daqing Chen |
| 라이선스 | CC BY 4.0 (UCI 페이지 표기, 2026-10-20 확인) |
| 인용 | Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |
| 다운로드 URL | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| 인증 | 없음 (인증키·계정·로그인 불필요) |
| zip 크기 | 45,622,418 bytes |
| 압축 해제 | `online_retail_II.xlsx` 45,622,278 bytes (xlsx 가 이미 압축 포맷이라 zip 이 거의 안 줄어든다) |
| zip SHA-256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| 확인 일자 | 2026-10-20 |
| 원본 행 수 | 1,067,371 (시트 2개: 525,461 + 541,910) |

## 재현 방법

```bash
python src/download_data.py      # 받기 + zip 검증 + 해시 출력
```

받은 파일의 크기가 위 값과 다르면 `download_data.py` 가 경고만 내고 멈추지 않음.
공공 데이터는 갱신될 수 있으므로 그때는 이 문서의 해시와 크기를 갱신할 것.

## 알려진 데이터 특성 (분석에 영향을 준 것)

- **두 시트가 9일 겹침:** `Year 2009-2010` 과 `Year 2010-2011` 이 2010-12-01 ~ 12-09 를 둘 다 담고 있음(45,046행, 송장 1,088건이 통째로 중복).
  단순 concat 하면 그 기간 매출 £754,977 중 절반이 허수 → `src/db.py` 의 UNIQUE 로 차단.
- **취소 전표:** 송장 번호가 `C` 로 시작하는 19,494행(원본 기준)이 있음.
  문자열로 읽지 않으면 parquet 저장에서 죽기 때문에 `build_db.py` 에서 dtype 고정.
- **고객 ID 결측:** 원본의 22.8%(적재 후 22.76%)가 고객 식별자 없음.
  이탈은 고객 단위 상태라 분석에서 제외했고, 매출의 13.1% 에 해당함.
- **시기:** 2009-12 ~ 2011-12 데이터라 최신 소비 행태와 다름.
  이 프로젝트의 산출물은 숫자가 아니라 파이프라인과 방법론임.
