#include <stdlib.h>
#include <stdio.h>

float *transpose(float *m, int row, int col) {
    float *c = malloc (sizeof(float) * row * col);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            c[j * row + i] = m[i * col + j];
        }
    }

    return c;
}

// test
/*int main() {
    float test[6] = {1, 2, 3, 4, 5, 6};
    float *t = transpose(test, 2, 3);

    for (int i = 0; i < 6; i++) {
        printf("%.2f ", t[i]);
    }
    printf("\n");
    return 0;
}*/