/* 이탈 예측 파이프라인 (C 구현) - 2단계: CSV 읽기 시험 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "csv.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif

#define LINE_SIZE 1024
#define DEFAULT_PATH "data/processed/export_for_c/clean_transactions.csv"

int main(int argc, char *argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    /* 인자로 경로를 주면 그 파일을 사용 */
    /* 없는 경우 기본 경로를 읽도록 함 */
    const char *path = (argc > 1) ? argv[1] : DEFAULT_PATH;
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: %s\n", path);
        return 1;
    }

    char line[LINE_SIZE];
    char *fields[CSV_MAX_FIELDS];

    /* 첫 줄: 열이름 */
    csv_read_line(fp, line, sizeof(line));
    int n = csv_split(line, fields, CSV_MAX_FIELDS);
    printf("열 %d개:", n);
    for (int i = 0; i < n; i++) {
        printf(" %s", fields[i]);
    }
    printf("\n");

    long long rows = 0;
    double revenue = 0.0;
    clock_t start = clock();

    int rc;
    while ((rc = csv_read_line(fp, line, sizeof(line))) == 1) {
        n = csv_split(line, fields, CSV_MAX_FIELDS);
        if (n != 7) {
            printf("필드 수가 7이 아님: %d행, %d개\n", (int)(rows + 2), n);
            fclose(fp);
            return 1;
        }

        long quantity = strtol(fields[4], NULL, 10);
        double price = strtod(fields[5], NULL);
        revenue += quantity * price;
        rows++;

        if (rows == 1) {
            printf("첫 행: 고객=%s 송장=%s 상품=%s 일시=%s 수량=%ld 단가=%.2f 국가=%s\n",
                   fields[0], fields[1], fields[2], fields[3], quantity, price,
                   fields[6]);
        }
    }
    fclose(fp);

    if (rc < 0) {
        printf("줄이 %d 바이트보다 길어서 잘림: %lld행 근처\n", LINE_SIZE,
               rows + 2);
        return 1;
    }

    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("읽은 행: %lld\n", rows);
    printf("매출 합계: %.2f\n", revenue);
    printf("걸린 시간: %.2f\n", seconds);
    return 0;
}
