# 3D Human Pose Reconstruction via Wieberg Apprach

This is the code for the Pose Estimation experiments
of the paper "Wibergian Learning for Continuous Energy 
Functions", by Matteo Toso, Neill Campbell and Chris Russell.
In NeurIPS 2019.

Dependencies

- h5py
- numpy
- tensorflow

## Set Up

The script 'set' will download the Human3.6M and stacked hourglass detections,
extract the necessary information and store them as training and testing 
data sets in the directory 'data'. This directory also contains the camera
parameters ('avg_cameras.h5'), the **PPCA** model parameters from *Tome et.al.*
(*'model_parameters.h5'*) and our trained model 'trained_model.h5'. 

## Files Content

The code here provided mostly coincides with the one of *Tome et.al.*, modified 
to allow for gradient propagation trough the whole **Tensorflow** graph.
1. *utils/draw.py* :: functions for plotting 2D detections and 3D poses;
2. *utils/math_operations.py* :: various mathematical function used by the main code;
3. *utils/parameters_io.py* :: functions to load and randomize training and testing 
set, and to load and save the model's parameters;
4. *utils/settings.py* :: contains the possible flags to customize the experiments,
and their default values;
5. *utils/wieberg.py* :: contains the class used to obtain reconstructions from 2D pose
detections;
6. *utils/train.py* :: contains the **Tensorflow** code to build the graph and process
the randomised training set, while updating the value of all trainable parameters.
7. *utils/test.py* :: contains the **Tensorflow** code to build the graph to evaluate the 
testing set, action by action.
8. *utils/config.py* :: provides paths used by various functions;

## Testing the code

The Python file *'Sample.py'* shows how the pose reconstruction works:
starting from four 2D poses, it generates the corresponding 3D reconstruction 
and compares it to the available ground truth.
To reproduce the results of our best trained model, launch the command

> python main.py --name='new_model' --check_path='data/trained_model.h5'

## Training and evaluating   

To train the model from scratch, with default settings and training all available parameters, 
launch the main python file specifying a name for the new project: 

> main.py --train=1 --name='new_model'

To customise the experiment (number of epochs, step size, size of training batches and so on),
edit the file 'utils/settings.py' or add one of the flags listed there while executing the
code. The initialisation function in file 'utils/wieberg.py' contains all parameters of
our problem; to keep any of them constant, just set it to 'trainable=False'. 
 
The flag *'--name'* is used to create a directory, in 'results',
to save all checkpoints of the new model and the training logs. 
The state of training can be monitored via **Tensorboard** 
(cd to 'new_model' and run *'tensorboard --log_dir logger'*).

To evaluate a trained model, launch *'main.py'* without the flag *'train=1'*,
and specifying the model parameters to load via the flag *'--check_path'*:

> main.py --wlr=5e-2 --name='new_model' --check_path='results/new_model/check/partial_checkpoint.h5'

If no model parameters are provided, the program will use the default values by
Tome et.al. and, where that is not possible, use the identity elements of the 
operation the parameters are used in.
