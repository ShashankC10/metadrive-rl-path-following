# Running on TAMU HPRC Grace (optional)

Source: HPRC "Introduction to Grace" primer (Sept 2025). Check the Grace user guide
(https://hprc.tamu.edu/kb/User-Guides/Grace/) before relying on anything here. The commands below
have **not** been run on Grace yet.

The core experiments do not need a cluster: one 2M-step PPO run takes roughly 10 minutes on an
Apple M3 Pro with 8 simulator workers. Grace is useful for the optional nuScenes / nuPlan phase
(large datasets, ScenarioNet conversion) or for many extra seeds.

## Facts from the primer that affect this project

| Topic | Grace |
| --- | --- |
| Hardware | Intel Cascade Lake, 48 cores and 384 GB per general compute node; A100/RTX 6000/T4 GPU nodes exist but are not needed (PPO uses `device="cpu"`) |
| Access | `ssh <NetID>@grace.hprc.tamu.edu` (VPN if off campus, two-factor auth) or the browser portal `portal.hprc.tamu.edu` |
| `$HOME` | 10 GB, 10,000 files. **Too small for the conda env and datasets** |
| `$SCRATCH` | 1 TB, 250,000 files, not backed up. Put the repo, conda env, datasets and checkpoints here; copy results you want to keep elsewhere. Check with `showquota` |
| Cost | 1 SU = 1 core-hour. One 10-min run on 8 cores is about 1.5 SU, so all six training runs are roughly 10 SU. Check `myproject` for your balance and default account |
| Queues | `short` (2 h), `medium` (1 day), `long` (7 days); assigned from the requested time. A 2M-step run fits in `short` |
| Jobs | Never run training on a login node. Submit with `sbatch`, monitor with `squeue --me`, cancel with `scancel`, check efficiency with `seff <jobid>` |
| Software | Modules (`module spider <name>`, `module load ...`, `module purge`). The primer's example loads `GCCcore/14.2.0 Python/3.13.1`, but MetaDrive 0.4.3 needs Python **3.11.x** |

## One-time setup (on a login node)

```bash
cd $SCRATCH
git clone <this repo> && cd metadrive-rl-path-following
module spider Miniconda3      # or Anaconda/Miniforge; if none fits, install Miniconda under $SCRATCH
# then, as in the README, with the env prefix under $SCRATCH to avoid the 10 GB home quota:
conda create --prefix $SCRATCH/envs/deep-rl python=3.11.9 pip -y
conda activate $SCRATCH/envs/deep-rl
python -m pip install -r requirements.txt && pip install -e .
python -m metadrive.pull_asset     # needs internet: do it here, not inside a job
python -m pytest -q                # confirm headless MetaDrive works on Linux before submitting jobs
```

## Job script

Based on the primer's template (`#SBATCH` lines are directives, not comments). Replace the account
number from `myproject`.

```bash
#!/bin/bash
#SBATCH --export=NONE
#SBATCH --get-user-env=L
#SBATCH --job-name=pf-ppo
#SBATCH --time=01:00:00
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=stdout.%j
#SBATCH --account=<ACCOUNT_NUMBER>

module purge
source activate $SCRATCH/envs/deep-rl     # `module load` your conda module first if needed
cd $SCRATCH/metadrive-rl-path-following
export PYTHONPATH=$PWD/src
python -m path_following.train --run ppo_s${SEED} --seed ${SEED} --resume
```

Submit one job per seed: `sbatch --export=ALL,SEED=0 job.slurm`. `n_envs` in the config is 8, which
matches `--cpus-per-task=8`; change both together.

## For the nuScenes / nuPlan phase

Download and convert under `$SCRATCH`, in a **separate** conda env (ScenarioNet and nuscenes-devkit
may conflict with the pinned core stack). Mind the 250,000-file limit: nuPlan and full nuScenes
contain very many small files, so check `showquota` while extracting. Compute nodes may not have
internet access, so do downloads on a login node or via the portal.
