#ifndef DATETIME_H
#define DATETIME_H

#include <stddef.h>

/* 하루의 초 */
#define SECONDS_PER_DAY 86400LL

/* 'YYYY-MM-DD HH:MM:SS' 를 1970-01-01 00:00:00 부터 지난 초로 바꿈 (시간대와 무관함) */
/* 돌려주는 값: 1 = 성공, 0 = 형식이 맞지 않거나 없는 날짜 */
int dt_parse(const char *text, long long *seconds);

/* 초를 'YYYY-MM-DD HH:MM:SS' 글자로 되돌려 buf에 사용함 */
void dt_format(long long seconds, char *buf, size_t size);

/* later 와 earlier 사이의 일수 */
/* 파이썬 Timedelta.days와 동일하게 소수점 이하는 내림 */
long long dt_days_between(long long later, long long earlier);

#endif
