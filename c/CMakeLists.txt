# 본격적으로 C 코드를 작성하기 전
# 컴파일러와 빌드 도구가 제대로 동작하는지 확인
# 이후 모든 파일을 이 빌드 구조 위에 하나씩 추가할 것

cmake_minimum_required(VERSION 3.20)
project(churn C)

# C11 표준을 사용함
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 실행 파일 churn 을 src/main.c 로 생성
add_executable(churn src/main.c)

# 경고를 많이 줘서 실수를 일찍 잡도록 함
target_compile_options(churn PRIVATE -Wall -Wextra)
