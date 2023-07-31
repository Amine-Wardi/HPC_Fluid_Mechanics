# High-Performance Computing: Fluid Mechanics with Python

This repository contains all the code and the report for this project, that was done for a lecture in the University of Freiburg.

There different experiments that are implemented in this project are : Shear Wave Decay, Couette Flow, Poiseuille Flow and Sliding Lid.

You can discover all the essential code for each milestone in the **src** folder. The code has been organized such that all functions reside within the `utils.py` file. Additionally, the code for plotting the figures associated with each milestone is contained within the corresponding milestone files. All the figures can be found in the **figures** directory.


## Installation

To run the code, all the libraries in the ```requirements.txt``` file need to be installed.

```
pip install -r requirements.txt
```


## Running the code

All the experiments are only coded sequentially, except for the sliding lid experiment, which is also implemented in a parallel manner as part of milestone 7.

### Serial

To run the code in serial, run the following command :

```
python3 src/file_name
```

Where file_name can be one the milestone files.

### Parallel

To run the sliding lid experiment in parallel, run the following command :
```
mpirun -n 4 python3 src/milestone7.py 
```

The number after -n is the number of processors.

The following arguments can be added at the end :
```
--ts : Time steps for the simulation. Default value is 100000
--gs : Grid size. Default value is (300, 300)
```