#include "csv.h"

#include <string.h>

int csv_read_line(FILE *fp, char *line, size_t size) {
    /* 한 줄을 읽음 */
    /* 읽을 것이 없는 경우 파일을 끝냄 */
    if (fgets(line, (int)size, fp) == NULL) {
        return 0;
    }

    size_t len = strlen(line);

    /* 줄바꿈이 없는데 파일 끝도 아닌 경우 줄이 버퍼보다 길어서 잘린 것으로 판단함 */
    if (len > 0 && line[len - 1] != '\n' && !feof(fp)) {
        return -1;
    }

    /* 끝의 \n, \r 제거 */
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[--len] = '\0';
    }
    return 1;
}

int csv_split(char *line, char *fields[], int max_fields) {
    int n = 0;
    char *p = line;

    while (n < max_fields) {
        /* 현재 위치를 새 필드의 시작으로 지정함 */
        fields[n++] = p;

        char *comma = strchr(p, ',');
        if (comma == NULL) {
            break;
        }

        /* 쉼표를 문자열 끝으로 바꿔서 앞 필드를 끊고 다음 필드로 이동함 */
        /* p = comma + 1로 포인터를 한 칸 앞으로 옮기는 방식으로 계산 */
        *comma = '\0';
        p = comma + 1;
    }
    return n;
}
