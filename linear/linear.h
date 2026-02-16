// linear.h
#ifndef LINEAR
#define LINEAR

typedef struct {
    float *weights;
    float bias;
    int input_dim;
    int output_dim;
} Linear;

Linear* create_linear(int input_dim, int output_dim);
float* linear_forward(Linear *linear, float *x, int n);

#endif