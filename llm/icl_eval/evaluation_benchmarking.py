

import pandas as pd
import sys
from mcli import sdk as msdk
import pathlib
import re
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from throughput.parse_logs import filter_runs

def get_runs(run_names):
    all_runs = [r for r in msdk.get_runs()]
    target_runs = []
    for target_run_name in run_names:
        target_runs.extend(
            [r for r in all_runs if r.name.startswith(target_run_name)]
        )
    return target_runs

def parse_logs(logs):
    lines = ''

    for line in logs:
        lines += line
    lines = lines.split('\n')

    run_time = None
    accuracy = None
    lines.reverse()
    for line in lines:
        match = re.search('Ran eval in: ((\d+)\.?(\d*)) seconds', line)
        if match and not run_time:
            run_time = float(match.group(1))

        match = re.search('metrics/.*?/.*?Accuracy: ((\d+)\.?(\d*))', line)
        if match and not accuracy:
            accuracy = float(match.group(1))

    
    return {
        "run_time": run_time,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    tsv_path = sys.argv[1]
    with open(tsv_path, "r") as f:
        df = pd.read_csv(f, sep='\t')
    runs = get_runs(list(df.run_name))
    runs = filter_runs(runs)
    for run in runs:
        logs = msdk.get_run_logs(run)
        result = parse_logs(logs)
        print(f"{run.name}: {result}")

