"""Run the 200-question experiment, validate results, then create statistics and plots."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / 'runs/qwen06b_gsm8k200_20rollouts'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # Control files live outside the runner's destination, which must start empty.
    control = ROOT / 'runs/qwen06b_control'
    control.mkdir(parents=True, exist_ok=True)
    lock = control / 'pipeline.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit('Pipeline lock exists. Check the recorded PID before restarting.')
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid()))
    state = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                 expected_rollouts=12000, results_directory=str(DESTINATION))
    def status(stage, **extra):
        state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = control / 'status.tmp.json'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(control / 'status.json')
        print(json.dumps(state), flush=True)
    try:
        status('running')
        command = [sys.executable, '-u', 'sampled_rlvr_experiment.py',
                   '--experiment-config', 'configs/qwen06b_gsm8k200.json',
                   '--limit', '200', '--rollouts', '20', '--output-dir', str(DESTINATION)]
        if args.resume:
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)
        status('validating_and_analyzing')
        subprocess.run([sys.executable, 'analyze_checkpoint_statistics.py', '--results-dir', str(DESTINATION),
                        '--input', 'data/gsm8k200/gsm8k200_injections.parquet', '--allow-missing-pairs'],
                       cwd=ROOT, check=True)
        status('plotting')
        subprocess.run([sys.executable, 'plot_checkpoint_means.py', '--results-dir', str(DESTINATION)], cwd=ROOT, check=True)
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
