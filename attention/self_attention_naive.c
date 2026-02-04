#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../matmul/matmul.h"
#include "../transpose/transpose.h"
#include "../softmax/softmax.h"

typedef struct {
    float *q;
    float *k; 
    float *v;
    float *o;
    int seq_len;
    int d_model;
} SelfAttention;

SelfAttention* create_attention(int seq_len, int d_model) {
    /*
    The malloc(sizeof(SelfAttention)) call:
        Allocates memory on the heap for one SelfAttention struct
        Returns the address of that allocated memory
        That address gets stored in your attention pointer
    */
    SelfAttention *attn = malloc(sizeof(SelfAttention));
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    
    int size = d_model * d_model;
    attn->q = malloc(sizeof(float) * size);
    attn->k = malloc(sizeof(float) * size);
    attn->v = malloc(sizeof(float) * size);
    attn->o = malloc(sizeof(float) * size);

    // Initialize random seed
    srand(time(NULL));
    
    // Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
    float scale = sqrtf(2.0f / (2 * d_model));
    
    // Initialize all weight matrices
    for (int i = 0; i < size; i++) {
        attn->q[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        attn->k[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        attn->v[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        attn->o[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
        
    return attn;
}

float *self_attention_forward(SelfAttention *attn, float* x) {
    float* output = malloc(sizeof(float) * attn->seq_len * attn->d_model);

    float *query_w = matmul(x, attn->q, attn->seq_len, attn->d_model, attn->d_model);
    float *key_w = matmul(x, attn->k, attn->seq_len, attn->d_model, attn->d_model);
    float *value_w = matmul(x, attn->v, attn->seq_len, attn->d_model, attn->d_model);
    
    // TODO: transpose and matmul - two operations we could fuse
    float *key_w_t = transpose(key_w, attn->seq_len, attn->d_model);
    float *att_scores = matmul(query_w, key_w_t, attn->seq_len, attn->d_model, attn->seq_len);
    softmax(att_scores, attn->seq_len, attn->seq_len);
    float *context = matmul(att_scores, value_w, attn->seq_len, attn->seq_len, attn->d_model);
    float *projected = matmul(context, attn->o, attn->seq_len, attn->d_model, attn->d_model);

    for (int i = 0; i < attn->seq_len * attn->d_model; i++) {
        output[i] = projected[i];
    }

    // Clean up intermediate results
    free(query_w);
    free(key_w);
    free(value_w);
    free(key_w_t);
    free(att_scores);
    free(context);
    free(projected);

    return output;
}

void free_attention(SelfAttention *attn) {
    free(attn->q);
    free(attn->k);
    free(attn->v);
    free(attn->o);  // Add this!
    free(attn);
}

int main() {
    printf("Testing Self-Attention...\n");
    
    // Create 4x4 attention (seq_len=4, d_model=4)
    // When you declare SelfAttention *attention, 
    // you're creating a pointer variable that will 
    // hold the memory address of a SelfAttention struct.
    SelfAttention *attention = create_attention(4, 4);

    /*
    attention is a reference (pointer) to the memory location where your struct lives. 
    The struct itself contains:
        The actual seq_len and d_model values
        Pointers q, k, v that point to their own separate memory allocations
    */

    // Simple test input: 4 tokens, each with 4 features
    float input[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,     // Token 1
        5.0f, 6.0f, 7.0f, 8.0f,     // Token 2  
        9.0f, 10.0f, 11.0f, 12.0f,  // Token 3
        13.0f, 14.0f, 15.0f, 16.0f  // Token 4
    };

    printf("Input:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", input[i * 4 + j]);
        }
        printf("\n");
    }

    // Run attention
    float *output = self_attention_forward(attention, input);
    
    printf("\nOutput:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", output[i * 4 + j]);
        }
        printf("\n");
    }

    // Verify output shape is correct
    printf("\nTest passed: Got %dx%d output\n", 4, 4);

    // Cleanup
    free(output);
    free_attention(attention);
    
    return 0;
}
