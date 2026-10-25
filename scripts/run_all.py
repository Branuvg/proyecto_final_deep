"""Run every stage in order with the default configs (works on CPU or GPU).
Usage (from the repo root, after `pip install -e .`): python scripts/run_all.py [--skip-notebooks]"""
import subprocess
import sys

STAGES = [
    ["-m", "pytest", "-q"],
    ["-m", "worldsim.data.generate", "--config", "configs/dataset.yaml"],
    ["-m", "worldsim.train.run_task_a", "--config", "configs/task_a.yaml"],
    ["-m", "worldsim.train.run_task_b", "--config", "configs/task_b.yaml"],
    ["-m", "worldsim.xai.run_xai", "--config", "configs/xai.yaml"],
    ["scripts/build_notebooks.py"],
]

if __name__ == "__main__":
    stages = STAGES[:-1] if "--skip-notebooks" in sys.argv else STAGES
    for s in stages:
        print(">>", "python", *s, flush=True)
        if subprocess.run([sys.executable, *s]).returncode:
            sys.exit(f"Falló la etapa: {' '.join(s)}")
