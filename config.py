import os

#General
cwd = os.getcwd()

#General
cwd = os.getcwd()


models_path = os.path.join(cwd, "models", "trained")
notebooks_path = os.path.join(cwd, "notebooks")
train_logs_path = os.path.join(models_path,"logs")

trained_model_path = models_path = os.path.join(models_path, "trained")

#data
data_path = os.path.join(cwd, "data")

data_raw = os.path.join(data_path, "raw")
data_processed = os.path.join(data_path, "processed")

#dev & test sets
data_raw_dev = os.path.join(data_raw, "dev")
data_raw_test = os.path.join(data_raw, "test")

data_processed_dev = os.path.join(data_processed, "dev")
data_processed_test = os.path.join(data_processed, "test")

