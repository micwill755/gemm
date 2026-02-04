#include <stdlib.h>
#include <stdio.h>

float* matmul(float *a, float *b, int m, int p, int n) {
    float *c = malloc(m * n * sizeof(float));

    for (int i = 0; i < m; i++){
        for (int k = 0; k < p; k++) {
            for (int j = 0; j < n; j++) {
                // c[i][j] += a[i][k] * b[k][j];
                /*
                     row-major ordering
                     indexing formula i * n + j for matrix c
                        - i * n skips to the start of row i
                        - j adds the column offset within that row
                    This stores matrix elements row-by-row in contiguous memory, which is the standard convention in C.
                */
                c[i * n + j] += a[i * p + k] * b[k * n + j];
            }
        }
    }

    return c;
}

// test
/*
int main() {
    int m = 4, p = 3, n = 4;

    float *a = malloc(m * p * sizeof(float));
    float *b = malloc(p * n * sizeof(float));

    // init matrices with random
    for (int i = 0; i < m * p; i++) {
        a[i] = rand() % 10;
    }
    for (int i = 0; i < p * n; i++) {
        b[i] = rand() % 10;
    }

    printf("A: \n");
    // print a and b
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", a[i * p + j]);
        }
        printf("\n");
    }

    printf("B: \n");
    // print a and b
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", b[i * n + j]);
        }
        printf("\n");
    }

    
    float *c = matmul(a, b, m, p, n);
    // print matrix c
    printf("C: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", c[i * n + j]);
        }
        printf("\n");
    }

    return 0;
}
*/