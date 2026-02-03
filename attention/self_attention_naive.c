#include <stdlib.h>
#include <stdio.h>
#include "../matmul/matmul.h"

typedef struct {
    long *q;
    long *k; 
    long *v;
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
    
    int size = seq_len * d_model;
    attn->q = malloc(sizeof(long) * size);
    attn->k = malloc(sizeof(long) * size);
    attn->v = malloc(sizeof(long) * size);
    
    return attn;
}

void self_attention_forward(SelfAttention *attn, long* x, long* output) {
    /* python:
        query_w = self.query.forward(x)
        key_w = self.key.forward(x)
        value_w = self.value.forward(x)

        att_scores = mat_mul(query_w, transpose(key_w))
        attn_weights = softmax(att_scores)
        context = mat_mul(attn_weights, value_w)
        context = self.out_proj.forward(context)
        
        return context
    */

    long *query_w = matmul(x, attn->q, attn->seq_len, attn->d_model, attn->d_model);
    long *key_w = matmul(x, attn->k, attn->seq_len, attn->d_model, attn->d_model);
    long *value_w = matmul(x, attn->v, attn->seq_len, attn->d_model, attn->d_model);
    
    // att_scores = matmul(query_w, transpose(key_w))
    // For now, simplified attention without transpose/softmax:
    // exercise 3 we will write transpose and softmax functions/kernels
    long *att_scores = matmul(query_w, key_w, attn->seq_len, attn->d_model, attn->seq_len);
    long *context = matmul(att_scores, value_w, attn->seq_len, attn->seq_len, attn->d_model);

    for (int i = 0; i < attn->seq_len * attn->d_model; i++) {
        output[i] = context[i];
    }
}

void free_attention(SelfAttention *attn) {
    free(attn->q);
    free(attn->k);
    free(attn->v);
    free(attn);
}

int main() {
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
    return 0;
}