#!/bin/bash

while getopts ":f:p:" opt; do
  case $opt in
    f) flow="$OPTARG"
    ;;
    p) ps="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ -z "$flow" ]
  then
    echo "No flow net specified"
    #exit -1
fi

if [[ $ps -eq 25 ]]
then
  patch_size=0.0652 #patch_size 25x25
elif [[ $ps -eq 51 ]]
then
  patch_size=0.1329 #patch_size 51x51
elif [[ $ps -eq 102 ]]
then
  patch_size=0.2657 #patch_size 102x102
elif [[ $ps -eq 153 ]]
then
  patch_size=0.3985 #patch_size 153x153
else
  echo "ERROR: Unknown patch size!"
  exit -1
fi

seeds=($(shuf -i 1-10000 -n 5))
first_run=false
for s in "${seeds[@]}"; do
    name_s="_s$s"
    name="atck_ps$ps$name_s"

    qsub -v seed=$s,patch_size=$patch_size,flownet=$flow -N $name run_attack.sh

    if $first_run ; then
        sleep 3m
        first_run=false
    fi
done

exit 0