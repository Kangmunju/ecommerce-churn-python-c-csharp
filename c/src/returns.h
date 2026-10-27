#ifndef RETURNS_H
#define RETURNS_H


#include <stddef.h>



/* 반품(취소) 전표 한 건 */
/* returns.csv 의 한 행 */
typedef struct 
{
  
    /* 1970-01-01 00:00:00 부터 지난 초 */
    long long timestamp;

    int customer_id;

    /* 맨 앞 C + 최대 6글자 + 끝 표시 */
    char invoice[8];
} Return;


/* 반품 전체를 담는 표 */
typedef struct 
{
    Return *items;
    size_t count;
    size_t capacity;
} ReturnTable;


/* CSV 를 읽어 table에 채움 */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
/* 실패 시 이유를 화면에 출력함 */
int returns_load(const char *path, ReturnTable *table);


/* table 이 쓰던 메모리를 돌려줌 */
void returns_free(ReturnTable *table);


#endif
