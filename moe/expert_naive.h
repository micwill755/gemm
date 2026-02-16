// expert.h
#ifndef EXPERT
#define EXPERT

typedef struct {
    float *weights;
    float bias;
    int emb_dim;
    int hidden_dim;
} Expert;

Expert* create_expert(int emb_dim, int hidden_dim);
float* expert_forward(Expert *expert, float *x, int n);

#endif