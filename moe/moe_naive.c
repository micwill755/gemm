#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "linear/linear.h"
#include "expert.h"
#include "softmax/softmax.h"

typedef struct {
    Linear *router;
    int top_k;
    int num_experts;
    int emb_dim;
    int seq_len;

    Expert **experts;
} MixtureOfExperts;

MixtureOfExperts* create_mixture_of_experts(int top_k, int num_experts, int seq_len, int emb_dim, int expert_dim) {
    MixtureOfExperts *mixtureOfExperts = malloc(sizeof(MixtureOfExperts));
    mixtureOfExperts->router = create_linear(emb_dim, num_experts);
    mixtureOfExperts->top_k = top_k;
    mixtureOfExperts->num_experts = num_experts;
    mixtureOfExperts->seq_len = seq_len;
    mixtureOfExperts->emb_dim = emb_dim;

    // create experts
    mixtureOfExperts->experts = malloc(num_experts * sizeof(Expert*));
    for (int e = 0; e < num_experts; e++) {
        mixtureOfExperts->experts[e] = create_expert(emb_dim, expert_dim);
    }
}

void top_k(float *expert_probs, int num_experts, int top_k, int *top_k_indices, float *top_k_probas) {
    for (int k = 0; k < top_k; k++) {
        float max_prob = -1.0f;
        int max_idx = 0;

        for (int i = 0; i < num_experts; i++) {
            int already_selected = 0;
            for (int j = 0; j < k; j++) {
                if (top_k_indices[j] == i) {
                    already_selected = 1;
                    break;
                }
            }
            
            if (!already_selected && expert_probs[i] > max_prob) {
                max_prob = expert_probs[i];
                max_idx = i;
            }
        }

        top_k_indices[k] = max_idx;
        top_k_probas[k] = max_prob;
    }
}

float* mixture_of_experts_forward(MixtureOfExperts *mixtureOfExperts, float*x, int num_batch, int n) {
    // step 1 - get expert values from input
    float *expert_scores = linear_forward(mixtureOfExperts->router, x, n);

    // step 2 - softmax
    float* expert_probs = malloc(sizeof(n));
    softmax(expert_probs, mixtureOfExperts->seq_len, n);

    // step 3 top k
    int* top_k_indices = malloc(mixtureOfExperts->top_k * sizeof(int));
    float* top_k_probs = malloc(mixtureOfExperts->top_k * sizeof(float));
    top_k(expert_probs, mixtureOfExperts->num_experts, mixtureOfExperts->top_k, top_k_indices, top_k_probs);

    float* output = malloc(num_batch * mixtureOfExperts->seq_len * mixtureOfExperts->emb_dim * sizeof(float));
    
    for (int b = 0; b < num_batch; b++) {
        for (int t = 0; t < mixtureOfExperts->seq_len; t++) {
            // create token embedding
            float* token_embedding = malloc(mixtureOfExperts->emb_dim * sizeof(float));
            for (int e = 0; e < mixtureOfExperts->emb_dim; e++) {
                token_embedding[e] = x[b * mixtureOfExperts->seq_len * mixtureOfExperts->emb_dim + t * mixtureOfExperts->emb_dim + e];
            }

            for (int k = 0; k < mixtureOfExperts->top_k; k++) {
                int e_idx = top_k_indices[k];
                float prob = top_k_probs[k];
                
                float* expert_output = linear_forward(mixtureOfExperts->experts[e_idx], token_embedding, mixtureOfExperts->emb_dim);
                
                for (int i = 0; i < mixtureOfExperts->emb_dim; i++) {
                    output[b * mixtureOfExperts->seq_len * mixtureOfExperts->emb_dim + t * mixtureOfExperts->emb_dim + i] += expert_output[i] * prob;
                }
            }
            
            free(token_embedding);
        }
    }
    
    free(expert_probs);
    free(top_k_indices);
    free(top_k_probs);
    
    return output;
}