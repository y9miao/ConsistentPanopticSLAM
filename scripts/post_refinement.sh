#!/bin/bash

SceneNum=$1
RefineResultFolder=$2
InitialResultFolder=$3


python scripts/semantic_graphcut_refine_tune.py \
--scene_num $SceneNum \
--result_folder $RefineResultFolder \
--base_folder $InitialResultFolder
