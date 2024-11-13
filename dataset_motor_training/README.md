This dataset contains 60 entries of initial and target end-effector positions couples and corresponding spikes produced as output by the motor cortex, as a sum of both the feedforward and feedback contributions. The recorded population is made up of a total of 100 neurons, equally split in two subgroups encoding positive and negative quantities, respectively. 
The dataset features 10 examples for each of the 6 combinations of start and final position, and each line is formatted as follows:

[initial_pos] [target_pos] [senders_pos] [times_pos] [senders_neg] [times_neg]
 
where senders contains the IDs of spiking neurons and times the instants at which spikes are generated.

Each line refers to a simulation starting with a 150 ms pause and preparation followed by 500 ms of movement execution.

The script utils.py contains some useful functions for the computation of inverse kinematics and dynamics, the planned, minimum jerk trajectory and the feedforward motor commands.
