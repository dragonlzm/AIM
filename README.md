# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

> **AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning** [[Paper](https://arxiv.org/abs/2412.03248)] <br>
> [Yiwu Zhong](https://scholar.google.com/citations?user=irrbH_IAAAAJ&hl=en)<sup>1</sup>, [Zhuoming Liu](https://scholar.google.com/citations?user=HXOVwRAAAAAJ&hl=en)<sup>2</sup>, [Yin Li](https://www.biostat.wisc.edu/~yli/)<sup>2</sup>, [Liwei Wang](https://lwwangcse.github.io/)<sup>#1</sup> <br>
> <sup>1</sup>The Chinese University of Hong Kong, <sup>2</sup>University of Wisconsin-Madison <br>
> (<sup>#</sup> corresponding author) <br>

<p align="center">
<img src="docs/teaser_figure.jpg" width=50% height=50%
class="center">
</p>


## Overview

Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that, our method substantially reduces computation load (e.g., a **7-fold** reduction in FLOPs) while preserving the performance of video and image LLMs. Further, under a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., **+4.6** on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs.


## Updates
- [12/4] 🔥 We release our paper. The code will be released soon. Stay tuned!


## Citation

If you find this repo useful, please consider citing our paper:

@article{zhong2024aim,
  title={AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning},
  author={Zhong, Yiwu and Liu, Zhuoming and Li, Yin and Wang, Liwei},
  journal={arXiv preprint arXiv:2412.03248},
  year={2024}
}


## Acknowledgement

- [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT): the pre-trained base Video LLM model.

- [LLaVA](https://github.com/haotian-liu/LLaVA): the pre-trained base Image LLM model.

- [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer): the code for calculating FLOPs and prefill time.

