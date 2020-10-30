#!/bin/bash
#PBS -N test_cuda
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1,mem=100mb,walltime=00:00:30
#PBS -m ae
#PBS -M schrodi@cs.uni-freiburg.de
#PBS -j oe
#PBS -q student

source ~/.bashrc
WORKDIR='/misc/student/schrodi/flowattack'
echo 'QSUB working on: $WORKDIR'
echo 'test'

nvidia-smi -q

exit 0
