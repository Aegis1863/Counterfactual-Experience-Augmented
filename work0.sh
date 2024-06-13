#!/bin/bash

#BSUB -q normal
#BSUB -o server_log/%J.out
#BSUB -e server_log/%J.out
#BSUB -a python3.9
#BSUB -n 1

export SUMO_HOME=~/anaconda3/envs/bowen-base/lib/python3.12/site-packages/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

~/anaconda3/envs/bowen-base/bin/python sumo_PPO.py -w 1
