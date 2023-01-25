

Launch sweep:
    `python3 -m common.mcli_sweep llm/icl_eval/sweep/eleuther_eval_benchmark_sweep.yaml llm/icl_eval/sweep/eleuther_eval_jobs.tsv`

Analyze sweep:
    `python3 llm/icl_eval/evaluation_benchmarking.py llm/icl_eval/sweep/eleuther_eval_jobs.tsv`