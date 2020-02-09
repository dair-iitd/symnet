
## Code organization

1. estimators.py: contains the main neural framework for policy and value module in the A3C algorithm.
2. worker.py: contains the code for initilizing the workers for the training of A3C. it copies the global network, runs the simulation and trains the network.
3. policy_monitor.py: contains the code for monitoring the policy on the instances after every time interval.
4. train.py: main script for specifying the parameters, staring the worker and the policy monitor.
5. transfer_test.py: contains the code for loading the previous trained weights and testing it on the new instances.
6. compile_results.py: script for compiling all the result in one csv after all the domains and instances have been trained and tested after transfer for easy evalution.
6. compile_results_baseline.py: script for compiling all the result in one csv after all the domains and instances have been trained from scratch on the larger instance for easy evalution.

## Training

To train the torpido agent on single instance use the following in this folder:

```python
python3 train.py --train_instance=5 --test_instance=5 --num_features=6 --neighbourhood=1 --model_dir=./train --domain=academic_advising --parallelism=4 --activation="lrelu" --lr=0.001 --num_gat_layers=1
```

Use the same instance number in train_instance and test_instance

#### Parameters

- train_instance: comman separated list of instances to train on
- test_instance: comman separated list of instances to validate/test on
- num_features: feature vector output of GAT
- neighbourhood: neighbourhood of aggregation of GAT
- model_dir: directory to save results to
- parallelism: number of worker threads in A3C
- activation: relu/lrelu/elu
- lr: learning rate
- domain: domain name to train on

#### Domain
1. academic_advising
2. crossing_traffic
3. game_of_life
4. navigation
5. skill_teaching
6. sysadmin
7. tamarisk
8. traffic
9. wildfire