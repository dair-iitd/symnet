# symnet

the requirements to use the experiments are
- python3
- tensorflow
- unittest
- multiprocessing
- threading
- shutil
- better_exceptions
- pickle
- networkx
- scipy

before starting the experiments run these commands from the main folder
```
for i in {1..30}
do
cp ./gym/envs/rddl/rddl/lib/clibxx.so ./gym/envs/rddl/rddl/lib/clibxx$i.so
cp ./rddl/lib/clibxx.so ./rddl/lib/clibxx$i.so
done
```
please see the individual folder on how to run the experiments
if you encounter C/C++ errors, please email