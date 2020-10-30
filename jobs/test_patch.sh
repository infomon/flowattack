#!/bin/bash
#PBS -N test_patch
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1,mem=1gb,walltime=01:00:00
#PBS -m ae
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
# cd $WORKDIR
# cd ../spatial_correlation_sampler-0.3.0
pip install spatial-correlation-sampler

cd $WORKDIR
python3 $WORKDIR/test_patch.py --name /misc/lmbraid19/schrodi/test --pretrained /misc/lmbraid19/schrodi/pretrained_models --patch_path patches/Upatch1.png

exit 0
