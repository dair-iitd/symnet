# SymNet
Public code for symnet for ICML submission


The requirements to use the experiments are:
1. python3
2. tensorflow=1.10
3. unittest
4. multiprocessing
5. threading
6. shutil
7. better_exceptions
8. pickle
9. networkx
10. scipy

Before starting the experiments run these commands from the main folder once:

```sh
for i in {1..10}
do
cp -r ./rddl/lib ./gym/envs/rddl/rddl
cp ./gym/envs/rddl/rddl/lib/clibxx.so ./gym/envs/rddl/rddl/lib/clibxx$i.so
cp ./rddl/lib/clibxx.so ./rddl/lib/clibxx$i.so
done
```
Add the following variables in the shell before running the code from the main folder:

```sh
MAIN_MODULE=$(pwd)

export PYTHONPATH=$MAIN_MODULE:$PYTHONPATH
export PYTHONPATH=$MAIN_MODULE/utils:$PYTHONPATH
```

Please see the individual folder on how to run the experiments

## Code organization

1. gym: contains the necessary interface for getting the states and actions during simulation of the planning problem. Also provides interface for taking the next step in the environment
2. rddl: contains the rddl domain files, parsed files and dbn files for all the domains for the experiment.
3. utils: contains the adapted code for the Graph Attention Networks (GAT)l, borrowed form [here](https://github.com/PetarV-/GAT).
4. multi_train: contains the code for the SymNet and the baseline  (adapted [ToRPIDo](https://github.com/dair-iitd/torpido) code).