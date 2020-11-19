#!/bin/bash
#PBS -N test_patch
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1,mem=3gb,walltime=00:30:00
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

cd $WORKDIR
python3 $WORKDIR/test_patch.py --name /misc/lmbraid19/schrodi --instance ps_25/lr1e3_552 --patch_name epoch_28 --flownet FlowNetC

exit 0
