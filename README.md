Methodology of making custom Qwen 3.5 397B EXL3 quants.

I've used Goldkoron's module sensivity table published by mratsim, converted it to JSON (`estimated_kld.json`), and generated override configs using `optimize_qwen_greedy_superlinear.py` script, then fed those configs to `mix_quants.py` and evaluated either using model_diff.py from exllamav3 repo or multi-gpu variant that splits work on many GPUs - both produce the same results and all published results, unless noted otherwise, use 100 rows.

Note that `optimize_qwen_greedy_superlinear.py` quants aren't quite on target and 3.5bpw input can come out as 3.6bpw mixed model.