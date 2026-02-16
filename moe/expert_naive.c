#include <stdlib.h>
#include <stdio.h>
#include "linear/linear.h"

typedef struct {
    Linear *fc1;
    Linear *fc2;
} Expert;

Expert* create_expert(int emb_dim, int hidden_dim) {
    Expert *expert = malloc(sizeof(Expert));
    expert->fc1 = create_linear(emb_dim, hidden_dim);
    expert->fc2 = create_linear(hidden_dim, emb_dim);
}

float* expert_forward(Expert *expert, float*x, int n) {
    x = linear_forward(expert->fc1, x, n);
    x = linear_forward(expert->fc1, x, n);
    return x;
}