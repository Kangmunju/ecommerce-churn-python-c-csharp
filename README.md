# 이커머스 고객 이탈 예측 (E-Commerce Customer Churn Prediction)

🌐 **[대시보드 바로가기](https://ecommerce-churn-python-c-csharp.streamlit.app/)** · 📄 **[한계 기록](LIMITS.md)** · 📄 **[가이드와 달라진 점](DEVIATIONS.md)**

거래 로그 106만 행으로 **"이 고객이 앞으로 6개월 안에 다시 살까"** 를 예측함.
성능 경쟁이 아니라, 이탈 예측에서 초보가 반드시 빠지는 함정을 실측으로 해체하는 프로젝트임.
같은 문제를 **Python(기준 구현) → C(초기 모델) → C#(객체지향 확장)** 으로 구현하고, 결과가 서로 맞는지 비교함.

## 어떻게 만들었나

- **Python 을 먼저 만들었음.** Python 구현을 먼저 끝내고 그 결과를 정답지로 삼았음. C 와 C# 은 그 정답지와 오차 범위 안에서 맞는지 비교하며 이어서 구현함. 커밋 순서가 작업 순서와 같음.
- **C(초기 모델):** 정제된 거래 CSV 에서 이탈 라벨, 피처 20개, 표준화, 로지스틱 회귀, 평가, 기대이익까지 라이브러리 없이 직접 구현함. 겪은 한계는 [LIMITS.md](LIMITS.md) 에 적음.
- **C#(객체지향 확장):** C 에서 부딪힌 한계(해시맵, 날짜, 제네릭, 시험 틀)를 풀고 교차검증, MLP, OOF 임계값, 확률 보정 확인, 가정 민감도를 추가함. xUnit 시험 115개가 붙어 있음.
- **AI 활용:** 구현 순서(Python → C → C#)와 설계, 코드 초안은 AI(Claude)와 함께 작성함. 실행과 검증은 직접 했고, 파이썬 정답지와 하나씩 대조해 어긋난 부분은 [DEVIATIONS.md](DEVIATIONS.md) 와 [LIMITS.md](LIMITS.md) 에 기록함.

## 한 줄 결론

| 무엇 | 숫자 |
|---|---|
| 대상 고객 | 4,944명 (관측창 2009-12-01~2011-06-09 에 1건 이상 구매) |
| 이탈률 | 48.0% (예측창 183일 재구매 0건) — 모집단과 창 길이를 같이 써야 의미가 있음 |
| 로지스틱 AUC | 0.7818 (5-fold 0.8044 ± 0.0080) |
| MLP AUC | 0.7886 (5-fold 0.8046 ± 0.0089) |
| **이탈 '정의'만 바꿨을 때** | **AUC 0.7818 → 1.0000** |

**MLP 가 로지스틱을 이겼다고 말하지 않음.** 시드 10개의 MLP AUC 는 0.7836 ± 0.0035 로 로지스틱(0.7818)과의 차이(+0.0018)가 시드 간 흔들림보다 작고, 이긴 시드는 10개 중 6개임.

## Python · C · C# 결과 비교

세 구현이 같은 시험 고객 1,484명에 낸 결과임. C# 의 `Churn.App` 이 24개 항목을 파이썬과 비교해 모두 허용 오차 안에서 일치함을 확인함.

| 항목 | Python(정답지) | C | C# |
|---|---|---|---|
| 로지스틱 AUC | 0.781834 | 0.781821 | 0.781821 |
| 혼동행렬 (TN / FP / FN / TP) | 527 / 245 / 181 / 531 | 526 / 246 / 180 / 532 | 526 / 246 / 180 / 532 |
| 최적 임계값 (시험 고객에서 선택) | 0.30 | 0.30 | 0.30 |
| 그때의 기대이익 | £6,969.34 | £6,969.34 | £6,969.34 |
| 임계값 0.50 의 기대이익 | £6,235.71 | £6,244.76 | £6,244.76 |
| OOF 로 고른 임계값 / 기대이익 | 0.25 / £6,908.90 | 구현 안 함 | 0.25 / £6,922.95 |
| 로지스틱 5-fold 평균 AUC | 0.8044 | 구현 안 함 | 0.8048 |
| MLP 5-fold 평균 AUC | 0.8046 | 구현 안 함 | 0.8015 |
| MLP 시드 10개 평균 AUC | 0.7836 | 구현 안 함 | 0.7857 |

- C 와 C# 이 파이썬과 1명씩 다른 혼동행렬과 0.50 의 기대이익은, **정답 CSV 의 계수가 sklearn 기본 허용 오차에서 덜 수렴한 값이기 때문**임. C·C# 의 계수는 충분히 수렴한 값(tol=1e-12)과 3.16e-06 이내로 일치함. → [DEVIATIONS.md](DEVIATIONS.md) 27번
- 교차검증 조각과 MLP 난수는 sklearn 과 달라서 조각별 값이 아니라 평균만 범위로 비교함. MLP 시드 평균은 파이썬보다 약 0.002 높고 원인은 확인하지 않음. → 28~30번
- 시간은 이 PC 에서 잰 참고값임. C 는 전체(MLP 와 교차검증 없음)가 약 1.1초, C# 은 MLP 를 뺀 범위가 약 1.8초(C 에 없는 교차검증 일부 포함)이고 MLP 를 포함한 전체는 약 13초임. C 는 최적화 옵션 없는 기본 빌드, C# 은 Release 빌드라 조건이 같지 않음. → [LIMITS.md](LIMITS.md)

## 이 프로젝트가 다루는 함정

| | 함정 | 실측 결과 |
|---|---|---|
| a2 | **이탈 '정의'** 를 바꾸면 | AUC 0.7818 → 1.0000. 라벨을 `recency` 로 만들면 모델이 정답을 베낌. recency 계열 8개를 지워도 0.8281 로 여전히 부풀려짐 |
| a3 | **시점 누수** — 피처를 전체 기간에서 만들면 | AUC 0.7829 → 0.9996 (+0.2167 이 공짜로 생김) |
| a4 | **생존편향** — 모집단을 바꾸면 | 이탈률 48.0% → 36.2%(1회 구매 고객 제외 시). 1회 구매 고객 이탈률은 73.9%. 단일 분할 AUC 는 오르지만 5-fold CV AUC 는 오히려 내려감 — 분할 노이즈를 개선으로 읽으면 안 됨 |
| a5 | **임계값은 돈이 정함** | 0.5 대신 0.30 에서 기대이익 £6,236 → £6,969 (임계값을 학습셋 out-of-fold 로 골라도 £6,909) |
| a6 | **MLP 를 표에 올릴 때** | 표준화를 빼면 MLP AUC −0.0697(로지스틱 +0.0024). 층을 60배 키워도 좋아지지 않음(0.780~0.790) |

## 실행 방법

모든 명령은 프로젝트 맨 위 폴더에서 실행함. Windows 11 에서 검증했고 다른 OS 는 검증하지 않음.

**1. Python (기준 구현, 데이터 준비 포함)**

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py            # 다운로드 → 적재 → 정제 → 라벨 → 피처 → 학습 → 분석 (약 2.5분, 첫 실행의 다운로드 제외)
```

- 필요 사양: 디스크 약 300MB, Python 3.14 에서 검증 (`requirements.txt` 는 검증한 버전으로 고정).
- 시간의 절반(약 80초)은 45MB 엑셀 파싱임. 한 번 parquet 으로 굳힌 뒤에는 쓰지 않음.
- 일부 단계만 다시 돌리려면 `python run_all.py --from 7` 또는 `--only 12`. 단계별 시간은 `reports/run_log.csv`.
- 모든 숫자는 `reports/*.csv` 로 자동 저장됨. 이 README 의 Python 숫자는 그 파일에서 옮긴 것임.

**2. C·C# 이 읽을 입력과 정답 만들기** (1번을 먼저 실행해야 함)

```powershell
python tools/export_for_c.py   # data/processed/export_for_c/ 에 CSV 7개 생성
```

**3. C**

```powershell
cmake -S c -B c/build -G "MinGW Makefiles"
cmake --build c/build
.\c\build\churn.exe            # 전체 파이프라인을 돌리고 파이썬 정답과 비교
```

- gcc(MinGW)와 CMake 3.20 이상이 필요함. 검증한 버전은 gcc 16.2, CMake 4.4.

**4. C#**

```powershell
dotnet run --project csharp/src/Churn.App -c Release   # 파이프라인 실행과 Python·C·C# 비교 표 (약 15초)
dotnet test csharp/Churn.sln                           # xUnit 시험 115개
```

- .NET SDK 9.0 이 필요함. 데이터 파일이 없으면 데이터를 쓰는 시험은 실패하지 않고 건너뜀.

**5. 대시보드 (Streamlit)**

```powershell
python tools/export_for_app.py                 # app/data/test_pred.csv 생성 (2번을 먼저 실행)
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

- 배포된 화면은 맨 위의 대시보드 링크에서 볼 수 있음. 모델은 다시 학습하지 않고 저장된 CSV 와 소스 코드를 읽어 보여 줌.
- 기대이익 계산기, 함정 4가지, 세 언어 비교(코드와 일치 확인), 방법과 한계의 4개 탭으로 구성됨.

## 데이터

UCI Machine Learning Repository — Online Retail II (CC BY 4.0)
영국 온라인 선물용품 소매업체의 실제 거래 원장. 2009-12-01 ~ 2011-12-09.
**인증키·로그인·계정 없이** `requests.get()` 한 번으로 받아짐. → [reports/DATA_ACCESS.md](reports/DATA_ACCESS.md)

## 폴더 구조

```
run_all.py                         Python 전체 파이프라인 (어느 위치에서 실행해도 동작)
LIMITS.md                          C 에서 부딪힌 한계와 C# 에서 확인한 것
DEVIATIONS.md                      원본 가이드·파이썬 정답지와 달라진 점 + 가이드 숫자 검증 결과
src/download_data.py               원본 다운로드 + 무결성 확인
src/build_db.py                    45MB 엑셀 → parquet (dtype 고정이 핵심)
src/db.py                          SQLite 적재. 중복은 UNIQUE 제약이 막음 (멱등)
src/clean.py                       정제 5단계 + 단계별 이력 기록
src/labels.py                      ★ 관측창/예측창 분리 이탈 라벨
src/features.py                    관측창만으로 만든 고객 피처 20개
src/train.py                       베이스라인 → 규칙 → 로지스틱 → MLP
src/plotstyle.py                   matplotlib 한글 폰트
analysis/a1_eda.py                 탐색
analysis/a2_label_definition.py    ★★ 이탈 정의
analysis/a3_leakage.py             ★★ 시점 누수
analysis/a4_survivorship.py        ★  생존편향
analysis/a5_profit.py              ★★ 기대이익 임계값
analysis/a6_mlp_pitfalls.py        ★  MLP 함정
tools/export_for_c.py              C·C# 입력과 파이썬 정답을 CSV 로 내보냄
tools/expected_values.py           C 시험의 기대값을 계산해 출처를 남김
tools/export_for_app.py            대시보드가 읽을 시험 고객의 정답과 예측 확률 CSV 를 만듦
app/streamlit_app.py               Streamlit 대시보드 (기대이익 계산기, 함정, 세 언어 비교, 한계)
app/data/                          대시보드 전용 작은 CSV (시험 고객 정답·확률, 비교 표)
.streamlit/config.toml             대시보드 색상 설정
c/CMakeLists.txt                   C 빌드 설정
c/src/                             csv, datetime, transactions, returns, labels, features,
                                   logistic, metrics, profit, main (라이브러리 없이 직접 구현)
csharp/Churn.sln                   C# 솔루션
csharp/src/Churn.Core/             Models, Data, Labels, Features, Modeling, Evaluation,
                                   Validation, Pipeline (로직)
csharp/src/Churn.App/              파이프라인 실행과 Python·C·C# 비교 표 출력
csharp/tests/Churn.Tests/          xUnit 시험 115개
reports/                           분석 결과 CSV·그림·데이터 접근 기록
```

## 한계 (숨기지 않음)

- 거래 데이터가 2009~2011 년임. 최신 소비 행태와는 다름. 파이프라인과 방법론이 이 프로젝트의 산출물이고, 숫자 자체는 그 시기의 것임.
- 고객 식별자가 없는 거래가 적재된 행의 22.8%(정제 5단계 투입 행 기준 22.6%)임. 이탈은 고객 단위 문제라 이 거래들은 분석에서 빠졌고, 그만큼(매출 13.1%)의 정보를 쓰지 못함.
- 모델 비교(로지스틱 vs MLP)는 test 1,484명 한 번의 분할과 5-fold CV 로 했음. AUC 의 표집 오차가 약 0.012 라서 0.01 안팎의 차이는 말할 수 없음.
- a5 의 기대이익은 쿠폰 반응률·마진·쿠폰 원가를 **가정**한 값임(유지 고객의 예측창 매출 중앙값만 실측). 쿠폰이 이탈을 실제로 되돌렸는지(증분 효과)는 이 데이터로 알 수 없음. 그건 실험이 필요함.
- MLP 수치는 라이브러리 버전·플랫폼에 따라 소수점 아래가 달라질 수 있음. 결론(로지스틱과 구분되지 않는다)은 시드를 바꿔도 같음.
- 학습/시험 분할은 파이썬의 무작위 분할 결과(`reference_split.csv`)를 C 와 C# 이 읽어 씀. 분할은 파이썬에 의존함.
- C·C# 구현에서 겪은 한계는 [LIMITS.md](LIMITS.md), 원본 가이드와 정답지와 달라진 점은 [DEVIATIONS.md](DEVIATIONS.md) 에 있음.
