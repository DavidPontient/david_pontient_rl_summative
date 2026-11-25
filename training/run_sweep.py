"""Run the whole hyperparameter sweep. Finished runs are skipped, so it is safe to stop and restart.

    uv run python -m training.run_sweep --algo all --workers 2
    uv run python -m training.run_sweep --algo ppo --runs 0-4
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys

from training.common import ROOT
from training.configs import SWEEPS


def parse_runs(text, n):
    if text is None:
        return list(range(n))
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def command(algo, run, seed, frames):
    if algo == "dqn":
        cmd = [sys.executable, "-m", "training.dqn_training", "--run", str(run), "--seed", str(seed)]
    else:
        cmd = [sys.executable, "-m", "training.pg_training", "--algo", algo, "--run", str(run), "--seed", str(seed)]
    if frames:
        cmd += ["--frames", str(frames)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="all", help="dqn, reinforce, ppo, a2c or all")
    parser.add_argument("--runs", default=None, help="for example 0-4 or 0,3,5")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    algos = list(SWEEPS) if args.algo == "all" else [args.algo]
    jobs = []
    for algo in algos:
        for run in parse_runs(args.runs, len(SWEEPS[algo][1])):
            done_file = os.path.join(ROOT, "logs", algo, f"run{run:02d}.json")
            if os.path.exists(done_file):
                print(f"skip {algo} run{run:02d} (already done)")
                continue
            jobs.append((algo, run))

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONWARNINGS="ignore", SDL_VIDEODRIVER="dummy")

    def work(job):
        algo, run = job
        print(f"start {algo} run{run:02d}", flush=True)
        subprocess.run(command(algo, run, args.seed, args.frames), cwd=ROOT, env=env)
        return job

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job in pool.map(work, jobs):
            print(f"finished {job[0]} run{job[1]:02d}", flush=True)


if __name__ == "__main__":
    main()
