#!/bin/bash -l
echo ${1}
echo ${2}

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate skeletal_alignment
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/pennaction_trainval.py"
main_cfg_path="configs/sa/sa_ds_penn.py"
ckpt_path="./"${2}

n_nodes=1
n_gpus_per_node=1
torch_num_workers=0
batch_size=1
pin_memory=true
val_steps=1
exp_name="CASA=$(($n_gpus_per_node * $n_nodes * $batch_size))"
#--benchmark=True \

python -u ./eval.py \
    --data_cfg_path=${data_cfg_path} \
    --main_cfg_path=${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=${val_steps} \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=${val_steps} \
    --limit_val_batches=1. \
    --num_sanity_val_steps=0 \
    --max_epochs=200 \
    --data_folder="./" \
    --ckpt_path=${ckpt_path} \
    --videos_dir=${videos_dir} \
    --dataset_name=${1}  