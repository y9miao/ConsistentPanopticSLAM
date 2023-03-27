#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate robot-scene-recon
python semantics/rp_server/launch_detectron_server.py