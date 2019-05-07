- To start the training use this command first

```sh
chmod +x ./*.sh
./run.sh 5
```

to start training 

```
./train.sh
```

to start transfer after the training is over

```
./retrain.sh
```

to edit parameters edit the following variables in the .sh file
- TI = training instance number (allowed numbers 1-10)
- TS = test instance number (allowed numbers 1-10)
- NF = number for features (allowed all ints)
- NR = neighbourhood (allowed all ints)
- LR = learning rate

allowd domains
- sysadmin
- academic_advising
- game
- if you want to run multiple instances at once use ',' separated numbers (eg. 5,12,17,10) in TI
- more instances you run at once, more time it will take to start the code

