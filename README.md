# Wibergian Learning

This is the code for the paper "Fixing Implicit Derivatives: Trust-Region Based
Learning of Continuous Energy Functions", by Matteo Toso, Neill Campbell and Chris Russell.
In NeurIPS 2019.

The code is divided into two directory: *RanSac* and *HumanPose*. The former 
contains the code used for the toy-model example provided in the paper,
and it is the best suited to understand how to implement our method for 
hyper-parameter tuning. 
The latter contains the code for the 3D Human Pose Reconstruction experiment,
heavily based on the ["Rethinking Pose in 3D"](https://arxiv.org/abs/1808.01525) paper.

# RanSac

Dependencies

- PyTorch
- python 3
- matplotlib

## Files Content

1. *wiberg.py* :: classes needed for the general Wibergian learning approach;
2. *toy.py* :: code used for the RanSac-like example;

## Running the program

By running

> python3 toy.py

you can run the toy experiment provided in the paper. This will generate a random
sed of values: 10 normally distributed 'true' values, and 100 uniformly distributed outliers.
We estimate the mean of the 'true' values as the minimum of an energy 
(the sum of 'RBF' functions), and we tune the energy hyper-parameters
by minimising the squared distance between the estimated and true mean. 
The code does this using our trust-region approach and an approach 
equivalent to just using implicit differentiation, and results for both cases
are plotted against each other.

As the true points and outliers are randomly generated, we suggest running the 
code multiple times.  

# 3D Human Pose Reconstruction

Dependencies

- python 2.7
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

## Trying the code

We first suggest trying our approach to human pose reconstruction. Running
 
> python Sample.py

will produce a 3D reconstruction starting from four 2D poses, 
compare it to the available ground truth.

The command

> python main.py --name='new_model' --check_path='data/trained_model.h5'

will reproduce the best results included in the paper, by running the whole
testing set trough the lifting process while using trained model's parameters.
This will create a new folder *'results/new_model'*, and produce *npy* files containing 
the reconstructed poses (*'poses.npy'*), the ground truth poses (*'truth.npy*),
the corresponding reconstruction errors (*'errors.npy'*). *'tabled_res.npy'*
contains the average per action reconstruction error, as reported in the paper's table. 

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
Tome et.al.


### Citing

If you use our code, please cite our work

```
@inproceedings{wibergianlearning_2019,
  title={Fixing Implicit Derivatives: Trust-Region Based Learning of Continuous Energy Functions},
  author={Toso, Matteo and Campbell, Neil and Russell, Chris},
  booktitle={NIPS 2019},
  year={2019}
}
```
