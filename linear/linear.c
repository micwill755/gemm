#include <stdlib.h>
#include <stdio.h>
#include "matmul/matmul.h"

typedef struct {
    float *weights;
    float bias;
    int input_dim;
    int output_dim;
} Linear;

Linear* create_linear(int input_dim, int output_dim) {
    Linear *linear = malloc(sizeof(Linear));
    linear->input_dim = input_dim;
    linear->output_dim = output_dim;
    linear->bias = 0.0f;

    int weights_size = input_dim * output_dim;
    linear->weights = malloc(weights_size * sizeof(float));

    for (int i = 0; i < input_dim; i++) {
        for (int j = 0; j < output_dim; j++) {
            linear->weights[i * input_dim + j] = (float)rand() / RAND_MAX * 1.0f;
        }
    }

    return linear;
}

// TODO: handle different tensor dimensions
float* linear_forward(Linear *linear, float *x, int n) {
    float *output = matmul(linear->weights, x, linear->input_dim, linear->output_dim, n);
    return output;
}