#!/bin/bash
#SBATCH --job-name=mypro
#SBATCH --output=log/mypro_%j.out
#SBATCH --error=log/mypro_%j.err
#SBATCH --array=1-20%5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd build

meshes=(100 200 500 1000)
mus=(0.10 0.15 0.20 0.30 0.33)

mesh_index=$((SLURM_ARRAY_TASK_ID / 5))
mu_index=$((SLURM_ARRAY_TASK_ID % 5))

mesh=${meshes[$mesh_index]}
mu=${mus[$mu_index]}

echo "Mesh = $mesh"
echo "Friction = $mu"
./steady_state "${mu}" "${mesh}"