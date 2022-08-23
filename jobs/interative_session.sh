
srun -p volta --pty -n 1 -c 4 --time=08:00:00 --gres gpu:1 --mem=16G bash -l
tmux new -s nested
module load volta anaconda3
conda activate
conda activate unsup_ctrl