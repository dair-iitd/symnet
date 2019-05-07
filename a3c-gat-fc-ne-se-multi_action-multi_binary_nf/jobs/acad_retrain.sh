#!/bin/zsh
#PBS -P ee

MAIN_MODULE=$(pwd)/../..

export PYTHONPATH=$MAIN_MODULE:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils/gat:$PYTHONPATH

JOB_DIR=$MAIN_MODULE/a3c-gat-fc-ne-se-multi_action-multi_binary_nf
cd $JOB_DIR

TI=5
NF=6
NR=0
LR=0.001

python3 transfer_train.py --train_instance=$TI --num_features=$NF --neighbourhood=$NR --model_dir=./retrain/acad --domain=academic_advising --num_instances=0 --parallelism=4 --activation="lrelu" --lr=$LR --restore_dir=$JOB_DIR/acad/academic_advising1-2-lrelu-6-20-20-0-0.001
