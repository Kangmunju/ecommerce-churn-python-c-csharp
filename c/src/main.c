/* 이탈 예측 파이프라인 (C 구현) - 1단계: 빌드 가능한지 확인 */

#include <stdio.h>
#include <time.h>

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif

int main(void) {
/* 윈도우에서 컴파일 할 때만 이 부분 포함 */
#ifdef _WIN32
    /* 한글이 깨지지 않도록 콘솔 출력 인코딩을 UTF-8(65001)로 지정함 */
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("churn (C 구현) 빌드 성공\n");
    printf("컴파일러: %s\n", __VERSION__);
    printf("크기(바이트): int=%zu long=%zu long long=%zu time_t=%zu double=%zu\n",
           sizeof(int), sizeof(long), sizeof(long long), sizeof(time_t),
           sizeof(double));
    return 0;
}
