#!/bin/bash
#PBS -N run_attack_lr1e3
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1,mem=3gb,walltime=23:59:00
#PBS -m a
#PBS -M schrodi@cs.uni-freiburg.de
#PBS -j oe
#PBS -q student

source ~/.bashrc
bash /misc/software/cuda/add_environment_cuda10.0.130_cudnnv7.5.sh
conda activate attack10.0_env

WORKDIR='/misc/student/schrodi/flowattack'
echo 'QSUB working on: $WORKDIR'

pip uninstall spatial-correlation-sampler -y
cd /home/schrodi/.cache/pip
find . -name '*spatial_correlation_sampler*' -delete
pip install spatial-correlation-sampler

patch_size=${patch_size}
echo "$patch_size"
echo "$flownet"
seed=${seed}
cd $WORKDIR
python3 $WORKDIR/main.py --name /misc/lmbraid19/schrodi --train-data /misc/lmbraid19/schrodi/KITTI/2012 --val-data /misc/lmbraid19/schrodi/KITTI/2012_val --valset kitti2012 --workers 1 --flownet $flownet --epochs 40 --patch-size $patch_size --seed $seed

exit 0
