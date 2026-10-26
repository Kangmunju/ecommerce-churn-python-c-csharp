/* 이탈 예측 파이프라인 (C 구현) */
/* 3단계 : 날짜 해석 시험 */

#include <stdio.h>
#include <string.h>

#include "csv.h"
#include "datetime.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 작성 */
#ifdef _WIN32
#include <windows.h>
#endif

#define LINE_SIZE 1024
#define DEFAULT_PATH "data/processed/export_for_c/clean_transactions.csv"

static int failures = 0;

/* 결과가 기대와 같은지 확인하고 출력함 */
static void check(const char *name, long long got, long long expected) {
    if (got == expected) {
        printf("  OK   %-46s %lld\n", name, got);
    } else {
        printf("  실패 %-46s 결과=%lld 기대=%lld\n", name, got, expected);
        failures++;
    }
}

/* 글자를 초로 바꿈, 실패하면 -1 */
static long long to_seconds(const char *text) {
    long long s;
    if (!dt_parse(text, &s)) {
        return -1;
    }
    return s;
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("[1] 알려진 값 (파이썬으로 구한 값과 비교)\n");
    check("1970-01-01 00:00:00", to_seconds("1970-01-01 00:00:00"), 0);
    check("2011-06-10 00:00:00 (기준일)", to_seconds("2011-06-10 00:00:00"),
          1307664000LL);
    check("2011-12-10 00:00:00 (데이터 끝 다음 날)",
          to_seconds("2011-12-10 00:00:00"), 1323475200LL);
    check("2009-12-01 07:45:00 (첫 거래)", to_seconds("2009-12-01 07:45:00"),
          1259653500LL);
    check("2012-02-29 23:59:59 (윤년)", to_seconds("2012-02-29 23:59:59"),
          1330559999LL);
    check("2000-02-29 00:00:00 (400년 윤년)", to_seconds("2000-02-29 00:00:00"),
          951782400LL);

    printf("[2] 잘못된 입력은 거절해야 함 (-1 이 기대값)\n");
    check("2011-02-29 00:00:00 (평년에 29일)", to_seconds("2011-02-29 00:00:00"), -1);
    check("2100-02-29 00:00:00 (100년은 평년)", to_seconds("2100-02-29 00:00:00"),
          -1);
    check("2011-13-01 00:00:00 (13월)", to_seconds("2011-13-01 00:00:00"), -1);
    check("2011-06-10 24:00:00 (24시)", to_seconds("2011-06-10 24:00:00"), -1);
    check("2011-06-10 (시각 없음)", to_seconds("2011-06-10"), -1);
    check("abc", to_seconds("abc"), -1);

    printf("[3] 일수 차이 (소수점 이하 내림)\n");
    long long cutoff = to_seconds("2011-06-10 00:00:00");
    check("기준일 - 2011-06-01 07:45:00", dt_days_between(cutoff, to_seconds("2011-06-01 07:45:00")), 8);
    check("기준일 - 기준일", dt_days_between(cutoff, cutoff), 0);
    check("데이터 끝 다음 날 - 기준일 (예측창)", dt_days_between(to_seconds("2011-12-10 00:00:00"), cutoff), 183);
    check("2011-06-01 - 기준일 (음수)", dt_days_between(to_seconds("2011-06-01 00:00:00"), cutoff), -9);
    check("2011-06-09 23:59:59 - 기준일 (음수 내림)", dt_days_between(to_seconds("2011-06-09 23:59:59"), cutoff), -1);

    printf("[4] 거래 파일의 모든 날짜를 해석하고 되돌려 비교\n");
    const char *path = DEFAULT_PATH;
    if (argc > 1) {
        path = argv[1];
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: %s\n", path);
        return 1;
    }

    char line[LINE_SIZE];
    char *fields[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));

    long long rows = 0, parse_fail = 0, round_trip_fail = 0;
    long long min_s = 0, max_s = 0, sum_s = 0;
    char back[32];

    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        csv_split(line, fields, CSV_MAX_FIELDS);
        long long s;
        if (!dt_parse(fields[3], &s)) {
            parse_fail++;
            continue;
        }
        dt_format(s, back, sizeof(back));
        if (strcmp(back, fields[3]) != 0) {
            round_trip_fail++;
        }
        if (rows == 0 || s < min_s) min_s = s;
        if (rows == 0 || s > max_s) max_s = s;
        sum_s += s;
        rows++;
    }
    fclose(fp);

    check("해석한 행 수", rows, 776577);
    check("해석 실패 행 수", parse_fail, 0);
    check("되돌린 글자가 원래와 다른 행 수", round_trip_fail, 0);
    check("가장 이른 거래의 초", min_s, 1259653500LL);
    check("가장 늦은 거래의 초", max_s, 1323435000LL);
    check("모든 거래의 초 합계", sum_s, 1004902583108760LL);

    if (failures == 0) {
        printf("\n모든 시험 통과\n");
        return 0;
    }
    printf("\n실패 %d건\n", failures);
    return 1;
}
