#!/bin/bash

# Run default eprop-reaching-task.py
python eprop-reaching-task.py

wait

python eprop-reaching-task.py --use-manual-rbf

wait

# Run eprop-reaching-task.py with scan parameters and two sets of trajectory files
python eprop-reaching-task.py \
  --trajectory-files ../dataset_motor_training/stage1/trajectories_90_to_140.txt,../dataset_motor_training/stage1/trajectories_90_to_20.txt \
  --target-files ../dataset_motor_training/stage1/spikes_from_90_to_140.txt,../dataset_motor_training/stage1/spikes_from_90_to_20.txt \
  --scan-param learning_rate,neurons.n_rec,task.n_iter,rbf.num_centers,synapses.w_input,synapses.w_rec,rbf.width \
  --scan-values "0.05;300;1000;20;200;30;0.1" \

  wait

  python eprop-reaching-task.py \
  --trajectory-files ../dataset_motor_training/stage1/trajectories_90_to_140.txt,../dataset_motor_training/stage1/trajectories_90_to_20.txt \
  --target-files ../dataset_motor_training/stage1/spikes_from_90_to_140.txt,../dataset_motor_training/stage1/spikes_from_90_to_20.txt \
  --use-manual-rbf \
  --scan-param learning_rate,neurons.n_rec,task.n_iter,rbf.num_centers,synapses.w_input,synapses.w_rec,rbf.width \
  --scan-values "0.05;300;1000;20;200;30;0.1" \