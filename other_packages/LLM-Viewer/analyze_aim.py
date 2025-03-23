import numpy as np
import ipdb

batchsize = 1
prank_iter = 5

# Vicuna
merge_iter = 3
vis_len = 576
feat_d = 1024

num_heads = 32
prompt_len = vis_len / 8 # 576 / 8
text_tokens = 40
retain_rate = [max(0, 1.0 - (layer_i - 13 + 1) * 0.125) if layer_i >= 13 else 1.0 for layer_i in range(32)] 

# Qwen2
merge_iter = 2
vis_len = 196 * 32
feat_d = 3584

num_heads = 28
prompt_len = vis_len / 4 # 196 * 32 / 4
text_tokens = 100
retain_rate = [max(0, 1.0 - (layer_i - 14 + 1) * 0.125) if layer_i >= 14 else 1.0 for layer_i in range(28)]


# token merging
def merging_ops_iter(batchsize, vis_len, feat_d):
    metric_OPs = (vis_len / 2) * (vis_len / 2) * feat_d * batchsize * 2
    norm_OPs = vis_len * feat_d * batchsize * 3
    merge_OPs = vis_len * feat_d * batchsize
    total_OPs = metric_OPs + norm_OPs + merge_OPs
    return total_OPs

def merging_ops(batchsize, vis_len, feat_d, merge_iter):
    total_OPs = 0
    for i in range(merge_iter):
        this_vis_len = vis_len // (2 ** i)
        OPs_this_iter = merging_ops_iter(batchsize, this_vis_len, feat_d)
        total_OPs += OPs_this_iter
        # print('Token merging: OPs this iteration (TB): ', round(OPs_this_iter / 10**12, 9))
    return total_OPs

# token pruning
def pruning_ops_layer(batchsize, prank_iter, num_heads, prompt_len):
    tau_softmax_OPs = batchsize * num_heads * prompt_len * prompt_len + batchsize * num_heads * prompt_len * prompt_len * 5 # softmax should have been already calculated by LLM
    prank_matmul_OPs = (1 * prompt_len) * prompt_len * num_heads * batchsize * 2 * prank_iter
    aggregation_OPs = (num_heads * prompt_len + num_heads * prompt_len + 1 * prompt_len) * batchsize
    total_OPs = tau_softmax_OPs + prank_matmul_OPs + aggregation_OPs
    return total_OPs

def pruning_ops(batchsize, prank_iter, num_heads, prompt_len, text_tokens):
    total_OPs = 0
    first_prune_OPs = pruning_ops_layer(batchsize, prank_iter, num_heads, prompt_len + text_tokens)
    total_OPs += first_prune_OPs
    # print('Token pruning: OPs at first pruning layer (TB): ', round(total_OPs / 10**12, 9))
    for rate in retain_rate:
        if rate > 0 and rate < 1:
            len_this_layer = int(prompt_len * rate) + text_tokens
            OPs_this_layer = pruning_ops_layer(batchsize, prank_iter, num_heads, len_this_layer)
            total_OPs += OPs_this_layer
            # print('Token pruning: OPs this layer (TB): ', round(OPs_this_layer / 10**12, 9))
    return total_OPs

# print('\nFirst merging: Flops (TB): ', round(merging_ops_iter(batchsize, vis_len, feat_d) / 10**12, 9))
merging_flops = merging_ops(batchsize, vis_len, feat_d, merge_iter)
print('Total merging: Total Flops (GB): ', round(merging_flops / 10**9, 9))

# print('\nFirst pruniing: Flops (TB): ', round(pruning_ops_layer(batchsize, prank_iter, num_heads, prompt_len + text_tokens) / 10**12, 9))
pruning_flops = pruning_ops(batchsize, prank_iter, num_heads, prompt_len, text_tokens)
print('Total pruning: Total Flops (GB): ', round(pruning_flops / 10**9, 9))

print('Total Flops (GB): ', round((merging_flops + pruning_flops) / 10**9, 9))


# Vicuna
# merging: 0.22708224
# pruning: 0.028326788
# total: 0.255409028

# Qwen2
# merging: 88.251957248
# pruning: 4.179420464
# total: 92.431377712

