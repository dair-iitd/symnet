#!/bin/zsh

MAIN_MODULE=$(pwd)/..

cd $MAIN_MODULE
MAIN_MODULE=$(pwd)
export PYTHONPATH=$MAIN_MODULE:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils/:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils/gat:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils/gat/:$PYTHONPATH

JOB_DIR=$MAIN_MODULE/a3c-gat-fc-ne-se-multi_action-multi_binary_nf
cd $JOB_DIR

TI=1
TS=2
NF=6
NR=0
LR=0.001

python3 train.py --train_instance=$TI --test_instance=$TS --num_features=$NF --neighbourhood=$NR --model_dir=./sys --domain=sysadmin --num_instances=0 --parallelism=4 --activation="lrelu" --lr=$LR


