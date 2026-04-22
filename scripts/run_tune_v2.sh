#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=2
#SBATCH -t 2-00:00:00
#SBATCH -o /projects/prjs1779/Osteosarcoma/output_ft/%x_%j.out
#SBATCH -e /projects/prjs1779/Osteosarcoma/output_ft_err/%x_%j.err
#SBATCH --job-name=OS-v2-res

# ----- Load modules -----
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

nvidia-smi --query-gpu=name,memory.total --format=csv

# ----- Activate virtual environment -----
source /projects/prjs1779/Osteosarcoma/.env/bin/activate

SCRIPT=/projects/prjs1779/Osteosarcoma/OS_CNN/src/main_tune_v2.py
SPLIT=/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/balance_datasplit/patient_splits.csv

# ── Fresh-start helper ─────────────────────────────────────────────────────────
# Optuna DB persists across jobs (load_if_exists=True), so re-submitting a job
# adds trials on top of previous ones.  Delete the DB for a fold before running
# to guarantee exactly --n_trials fresh trials.
#
# Base path:  /scratch-shared/xwan1/experiments/<EXP>/tune/<PREFIX>_<MODEL>/outer_fold_<K>/optuna_study.db
#
# Example — delete fold 0 DB before running:
# rm -f /scratch-shared/xwan1/experiments/tune_v2/tune/run_resnet/outer_fold_0/optuna_study.db

# ── Uncomment exactly ONE line below to run the desired fold ──────────────────

python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 0  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 1  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 2  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 3  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 4  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 5  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 6  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 7  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 8  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 9  --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 10 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 11 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 12 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 13 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 14 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 15 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 16 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 17 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 18 --n_trials 10 --random_seed 42 --split_file $SPLIT
# python $SCRIPT --modality T1W --version v1 --model_type resnet --experiment_name tune_v2 --prefix run --n_fold 19 --n_trials 10 --random_seed 42 --split_file $SPLIT
