import papermill as pm
from time import time, gmtime

# This file, all other imported .py files, and the blank notebooks should be visible in the same folder.
# Otherwise, some paths will have to be changed in various places.

# Run file with this command:
# nohup python run_models.py > multi_model_run.log &


BATCH_NUM = 3

start_time = time()

pm.execute_notebook(
      f'./Multi_Model_Building_(Batch_{BATCH_NUM}).ipynb',
      f'./trained_notebooks/Multi_Model_Building_(Batch_{BATCH_NUM})_trained.ipynb'
   )
total_time = time() - start_time
conv_time = gmtime(total_time)
print(f'Training finished in: {conv_time.tm_hour} hours, {conv_time.tm_min} minutes, {conv_time.tm_sec} seconds')

# PID: XXXX