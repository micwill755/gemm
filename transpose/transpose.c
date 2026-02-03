#include <stdlib.h>
#include <stdio.h>

long *transpose(long *m, int row, int col) {
    long *c = malloc (sizeof(long) * row * col);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            c[j * row + i] = m[i * col + j];
        }
    }

    return c;
}

int main() {
    long test[6] = {1, 2, 3, 4, 5, 6};
    long *t = transpose(test, 2, 3);

    for (int i = 0; i < 6; i++) {
        printf("%ld ", t[i]);
    }
    printf("\n");
    return 0;
}