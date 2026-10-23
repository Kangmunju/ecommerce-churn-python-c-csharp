#ifndef CSV_H
#define CSV_H

#include <stddef.h>
#include <stdio.h>

/* 한 줄에서 쪼갤 수 있는 최대 필드 수*/
#define CSV_MAX_FIELDS 32

/* fp 에서 한 줄을 읽어 line 에 저장함 */
/* 끝의 줄바꿈은 제거함 */
/* 돌려주는 값: 1 = 한 줄 읽음, 0 = 파일 끝, -1 = 줄이 size 보다 길어서 잘림 */
int csv_read_line(FILE *fp, char *line, size_t size);

/* line 안의 쉼표를 문자열 끝('\0') 으로 바꿔서 필드로 쪼갬 */
/* fields[i] 는 line 안의 i번째 필드를 가리킴 */
/* 쪼갠 필드 수를 리턴함 */
int csv_split(char *line, char *fields[], int max_fields);

#endif
