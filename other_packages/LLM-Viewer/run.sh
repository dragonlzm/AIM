# Scipt to calculate the prefill with different token numbers at different layers (Our Method, noted that Qwen2 has 28 layers, while the vicuna has 32 layers)
python3 analyze_flex_prefill_only.py Qwen/Qwen2-7B nvidia_A100 --config_file configs/Llama.py --skip-mlp-bias

# python3 analyze_flex_prefill_only.py lmsys/vicuna-7b-v1.5 nvidia_A100 --config_file configs/Llama.py 


# Scipt to calculate the prefill with same token numbers at different layers (LLaVA-PruMerge)
# python3 analyze_flex_prefill_only.py Qwen/Qwen2-7B nvidia_A100 --config_file configs/Llama.py  --promptlen 18532 --skip-mlp-bias

# python3 analyze_flex_prefill_only.py lmsys/vicuna-7b-v1.5 nvidia_A100 --config_file configs/Llama.py  --promptlen 328

# python3 analyze_flex_prefill_only.py Qwen/Qwen-VL-Chat nvidia_A100 --config_file configs/Llama.py  --promptlen 296  # Qwen-VL-Chat with 256 learnable queries