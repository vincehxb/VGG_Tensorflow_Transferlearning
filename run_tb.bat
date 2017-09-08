@echo off  
start cmd /k "echo open tensorboard&&activate py35&&echo log_dir=mylog&&tensorboard --logdir=./mylog" 