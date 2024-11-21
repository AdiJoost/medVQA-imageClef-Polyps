import os
from os.path import join

#General
cwd = os.getcwd()

#General
cwd = os.getcwd()


models_path = join(cwd, "models")
notebooks_path = join(cwd, "notebooks")
train_logs_path = join(models_path,"logs")

trained_model_path = models_path = join(models_path, "trained")

#data
data_path = join(cwd, "data")

data_raw = join(data_path, "raw")
data_processed = join(data_path, "processed")

#dev & test sets
data_raw_dev = join(data_raw, "dev")
data_raw_test = join(data_raw, "test")

data_raw_dev_images = join(data_raw_dev, "images")
data_raw_test_images = join(data_raw_test, "images")

data_processed_dev = join(data_processed, "dev")
data_processed_test = join(data_processed, "test")

