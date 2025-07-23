#!/bin/bash

# python eprop-reaching-task.py \
#   --trajectory-files /home/shimoura/Documents/GitHub/motor-controller-model-add-nestml-neurons/dataset_motor_training/stage1/trajectories_90_to_140.txt,/home/shimoura/Documents/GitHub/motor-controller-model-add-nestml-neurons/dataset_motor_training/stage1/trajectories_90_to_20.txt \
#   --target-files /home/shimoura/Documents/GitHub/motor-controller-model-add-nestml-neurons/dataset_motor_training/stage1/spikes_from_90_to_140.txt,/home/shimoura/Documents/GitHub/motor-controller-model-add-nestml-neurons/dataset_motor_training/stage1/spikes_from_90_to_20.txt \
#   --use-manual-rbf \
# #   --no-plot

# python eprop-reaching-task.py --use-manual-rbf
# wait
# python eprop-reaching-task.py

# wait

# python eprop-reaching-task.py --use-manual-rbf --plastic-input-to-rec --learning-rate 0.1
# wait
# python eprop-reaching-task.py --plastic-input-to-rec --learning-rate 0.1

python eprop-reaching-task.py --use-manual-rbf --scan-param neurons.n_rec,rbf.num_centers --scan-values "200;15,20"
wait
python eprop-reaching-task.py --scan-param neurons.n_rec,rbf.num_centers --scan-values "200;15,20"