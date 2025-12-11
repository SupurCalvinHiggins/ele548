import json
from pathlib import Path
import torch

for metrics_path in Path("results-256").glob("**/metrics.json"):
    cfg_path = metrics_path.parent / "cfg.json"
    metrics = json.loads(metrics_path.read_text())
    print(metrics_path.parent)
    print(cfg_path.read_text())
    ifs = []
    for benchmark_uri, benchmark_metrics in metrics.items():
        print(benchmark_uri, benchmark_metrics["mean_rl_oz_improvement_factors"][-1], benchmark_metrics["min_rl_oz_improvement_factors"][-1], benchmark_metrics["max_rl_oz_improvement_factors"][-1])
        ifs.append(benchmark_metrics["mean_rl_oz_improvement_factors"][-1])
    print(torch.tensor(ifs).log().mean().exp().item())
    print()