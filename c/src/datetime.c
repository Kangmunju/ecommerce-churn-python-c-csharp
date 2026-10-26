#include "datetime.h"

#include <stdio.h>

/* 윤년이면 1로 표기함 */
static int is_leap(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

/* 그 해의 날짜 수 (윤년이면 366, 아니면 365) */
static int days_in_year(int year) {
    if (is_leap(year)) {
        return 366;
    }
    return 365;
}

/* 그 해, 그달의 날짜 수 계산 */
static int days_in_month(int year, int month) {
    static const int days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && is_leap(year)) {
        return 29;
    }
    return days[month - 1];
}

/* 1970-01-01 부터 year-month-day 까지 지난 일수 */
/* 1970년 이후의 데이터만 받아올 수 있었음 */
static long long days_from_civil(int year, int month, int day) {
    long long days = 0;

    /* 1970년부터 전년도 말까지의 일수 계산 */
    for (int y = 1970; y < year; y++) {
        days += days_in_year(y);
    }

    /* 1월부터 전달 말까지의 일수 계산 */
    for (int m = 1; m < month; m++) {
        days += days_in_month(year, m);
    }

    /* 그 달의 일수 계산 */
    /* 1일이면 0일 지난 것으로 판단 */
    return days + (day - 1);
}

/* days_from_civil 의 반대 */
/* 지난 일수로부터 년, 월, 일을 계산함 */
static void civil_from_days(long long days, int *year, int *month, int *day) {
    int y = 1970;
    while (days >= days_in_year(y)) {
        days -= days_in_year(y);
        y++;
    }

    int m = 1;
    while (days >= days_in_month(y, m)) {
        days -= days_in_month(y, m);
        m++;
    }

    *year = y;
    *month = m;
    *day = (int)days + 1;
}

int dt_parse(const char *text, long long *seconds) {
    int year, month, day, hour, minute, second;

    /* 'YYYY-MM-DD HH:MM:SS' 형식을 만들기 위해 위 숫자 6개를 읽음 */
    if (sscanf(text, "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute,
               &second) != 6) {
        return 0;
    }

    /* 범위 확인 */
    /* 없는 날짜등은 거절함 */
    if (year < 1970 || month < 1 || month > 12) {
        return 0;
    }
    if (day < 1 || day > days_in_month(year, month)) {
        return 0;
    }
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 ||
        second > 59) {
        return 0;
    }

    *seconds = days_from_civil(year, month, day) * SECONDS_PER_DAY + hour * 3600 +
               minute * 60 + second;
    return 1;
}

void dt_format(long long seconds, char *buf, size_t size) {
    long long days = seconds / SECONDS_PER_DAY;
    long long rest = seconds % SECONDS_PER_DAY;

    int year, month, day;
    civil_from_days(days, &year, &month, &day);

    snprintf(buf, size, "%04d-%02d-%02d %02d:%02d:%02d", year, month, day,
             (int)(rest / 3600), (int)(rest % 3600 / 60), (int)(rest % 60));
}

long long dt_days_between(long long later, long long earlier) {
    long long diff = later - earlier;
    long long days = diff / SECONDS_PER_DAY;

    /* C의 나눗셈은 0쪽으로 자름 */
    /* 음수이고 나누어떨어지지 않는 경우 1을 더 내려줌 */
    if (diff % SECONDS_PER_DAY < 0) {
        days -= 1;
    }
    return days;
}

/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. strptime, mktime 을 쓰지 않은 이유 (이 컴퓨터에서 직접 확인함) */
/*    strptime: Windows(MinGW)에 없어서 컴파일 오류 (implicit declaration of function) */
/*    mktime: 컴퓨터의 시간대에 따라 결과가 달라짐 (한국 시간에서 2011-06-10 이 UTC 기준보다 -32,400초) */
/*    그래서 sscanf 로 숫자 6개를 읽고 1970-01-01 부터의 초를 직접 계산함 -> 시간대와 무관하게 같은 값 */
/* */
/* 2. 일수 차이는 내림으로 계산함 */
/*    파이썬 Timedelta.days 는 항상 내림이고 C 의 정수 나눗셈은 0 쪽으로 자름 */
/*    양수에서는 같지만 음수에서 다름 (-1초는 파이썬에서 -1일, C 나눗셈은 0일) */
/*    그래서 음수이고 나누어떨어지지 않으면 1을 더 내림 */
/* */
/* 3. 한계 */
/*    1970년 이후의 날짜만 지원함 (데이터는 2009 ~ 2011년) */
/*    연도를 하나씩 더하는 반복문이라 먼 미래의 날짜에는 느려질 수 있음 */
/*    초 값은 long long 사용 (모든 거래의 초 합계가 약 1e15 라서 int 로는 넘침) */
/* ==================================================================================================== */
