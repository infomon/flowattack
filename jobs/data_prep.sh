#!/bin/bash
#PBS -N data_prep
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,mem=1gb,walltime=24:00:00
#PBS -m ae
#PBS -M schrodi@informatik.uni-freiburg.de
#PBS -j oe
#PBS -q student

source ~/.bashrc
conda activate attack_env

WORKDIR='/misc/student/schrodi/flowattack'
echo 'QSUB working on: $WORKDIR'


python3 $WORKDIR/data/prepare_train_data.py /misc/lmbraid19/schrodi/KITTI/raw/ --dataset-format 'kitti' --dump-root /misc/lmbraid19/schrodi/KITTI/2012/ --width 1280 --height 384 --num-threads 1 --with-gt

exit 0
