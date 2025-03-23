# LLM-Viewer

The code is adapted from LLM-Viewer(https://github.com/hahnyuan/LLM-Viewer) with minimum modification on script interface.

Run `bash run.sh` to calculate the FLOPs and prefill time of our default inference configuration. 

The key file is `analyze_flex_prefill_only.py` where you can specify the number of initial visual tokens, the number of text tokens, and the scheduler to prune tokens across LLM layers.