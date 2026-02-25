#!/bin/bash
#SBATCH --job-name=bench_tsne
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -A m636
#SBATCH -o /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/bench_tsne_%j.out
#SBATCH -e /global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend/logs/slurm/bench_tsne_%j.err

set -euo pipefail
BACKEND_DIR="/global/cfs/cdirs/m636/nhadler/projects/Latent_Space_Vis/latentvis/backend"
cd "$BACKEND_DIR"

module load conda
conda activate latentvis

python3 - <<'PY'
import time
import numpy as np
from sklearn.manifold import TSNE
import sklearn

print("sklearn", sklearn.__version__)
n=113
d=128
X=np.random.RandomState(0).randn(n,d).astype(np.float32)

def run(max_iter):
    t0=time.time()
    TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42,
        learning_rate="auto",
        max_iter=max_iter,
    ).fit_transform(X)
    return time.time()-t0

for it in (250, 350, 500):
    times=[]
    for _ in range(3):
        times.append(run(it))
    print("max_iter", it, "secs", times, "avg", sum(times)/len(times))
PY

