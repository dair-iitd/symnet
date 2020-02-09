
## Code organization

1. symnet: contains the code for training symnet on small instances as well as transferring it to large instances. It also contains the code for training symnet from scratch on large instances for baseline.
2. torpido: contains the code for baseine torpido training from scratch on any instance.
3. tensorboard_to_csv: constains the code for converting the tensorboard readings to csv for easy result evaluation.


### Running the experiments

For running the symnet transfer experiment or the baseline training from scratch, refer to the readme in the symnet folder

For running the torpido baseline training from scratch, refer to the readme in the torpido folder


### tensorboard_to_csv usage

To convert any tensorboard log to csv, run the following:

```python
python3 tensorboard_to_csv.py
```

This will recursively convert in tensorboard readings to csv in all the sub folders and store in under val_csv folder in corresponding folder.

