#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void softmax(float *m, int row, int col) {
    // Process each row
    for (int i = 0; i < row; i++) {
        // Step 1: Find row max for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < col; j++) {
            max_val = fmaxf(m[i * col + j], max_val);
        }

        // Step 2: Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int j = 0; j < col; j++) {
            sum += expf(m[i * col + j] - max_val);
        }

        // Step 3: Normalize
        for (int j = 0; j < col; j++) {
            m[i * col + j] = expf(m[i * col + j] - max_val) / sum;
        }
    }
}

// test
/*
int main() {
    // Test 1: Simple 1x3 vector
    printf("Test 1: [1, 2, 3]\n");
    float test1[3] = {1.0f, 2.0f, 3.0f};
    softmax(test1, 1, 3);
    for (int i = 0; i < 3; i++) {
        printf("%.6f ", test1[i]);
    }
    printf("\nSum: %.6f\n\n", test1[0] + test1[1] + test1[2]);

    // Test 2: Large attention-like matrix (4x4)
    printf("Test 2: 4x4 attention scores\n");
    float attention_scores[16] = {
        0.5f, 1.2f, -0.3f, 2.1f,
        1.8f, 0.1f, -1.5f, 0.9f,
        -0.7f, 2.3f, 1.1f, -0.2f,
        0.8f, -1.1f, 1.9f, 0.4f
    };
    
    printf("Before softmax:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", attention_scores[i * 4 + j]);
        }
        printf("\n");
    }
    
    softmax(attention_scores, 4, 4);
    
    printf("\nAfter softmax:\n");
    for (int i = 0; i < 4; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < 4; j++) {
            printf("%.4f ", attention_scores[i * 4 + j]);
            row_sum += attention_scores[i * 4 + j];
        }
        printf("(sum: %.4f)\n", row_sum);
    }
    
    return 0;
}
*/