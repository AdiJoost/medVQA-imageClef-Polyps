import tensorflow as tf
import numpy as np
from keras import datasets, utils
import config
import os
import cv2
import matplotlib.pyplot as plt
import requests
import zipfile
import datetime
import pathlib
import shutil
import config
import pathlib
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

QUESTIONS_TO_BE_ANSWERED = [
    "What type of procedure is the image taken from?",
    "How many instrumnets are in the image?", #spelled wrong, but is like this in their test script too
    "Have all polyps been removed?",
    "Where in the image is the abnormality?",
    "Is this finding easy to detect?",
    "Where in the image is the instrument?",
    "Is there a green/black box artifact?",
    "Are there any abnormalities in the image?",
    "Is there text?",
    "Are there any anatomical landmarks in the image?",
    "What color is the abnormality?",
    "Are there any instruments in the image?",
    "What color is the anatomical landmark?",
    "Where in the image is the anatomical landmark?",
    "How many findings are present?",
    "What is the size of the polyp?",
    "How many polyps are in the image?",
    "What type of polyp is present?"
    ]


def main():
    # encode_dev_data()
    train_ds,_,_ = get_dev_datasets()
    print(train_ds)
    
def get_dev_datasets():
    """
    Returns the train, test, val datasets of the dev dataset in format
    ((image, question), answer_vector)
    """
    
    if not os.path.exists(os.path.join(config.data_raw_dev, "X_train.npy")):
        encode_dev_data()
        
    X_train, X_test, X_val, y_train, y_test, y_val = load_dev_data()
    
    train_ds = dataset_from_X_y(X_train, y_train)
    test_ds = dataset_from_X_y(X_test, y_test)
    val_ds = dataset_from_X_y(X_val, y_val)
    
    return train_ds, test_ds, val_ds


def dataset_from_X_y(X,y):
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    ds = ds.map(lambda x,y: (load_image_from_ID(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds.batch(batch_size=64)

def load_image_from_ID(X):
    """
    change the path here, to change, where the images are loaded from, later from processed folder
    """
    image_id = X[0]  
    question = X[1]  
    
    image_path = tf.strings.join([config.data_raw_dev, str(image_id) + ".jpg"], separator=os.path.pathsep)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = image / 255.0  # Normalize pixel values

    return (image, question)



def get_data():
    """
    Extracts the raw data from the zip files if it exists and puts them into the raw/(dev|test) folders
    """
    os.makedirs(config.data_raw_dev)
    os.makedirs(config.data_raw_test)
    raw_dev = pathlib.Path(config.data_raw_dev)
    files_raw_dev = list(raw_dev.glob("*.zip"))
    if len(files_raw_dev) == 1:
        dev_zip = files_raw_dev[0]
        unzip(dev_zip, config.data_raw_dev)
        
    raw_test = pathlib.Path(config.data_raw_test)
    files_raw_test = list(raw_test.glob("*.zip"))
    if len(files_raw_test) == 1:
        test_zip = files_raw_test[0]
        unzip(test_zip, config.data_raw_test)
    

def unzip(zip_file, destination):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"Extracted contents to the '{destination}' folder.")
    os.remove(zip_file)

    
def encode_dev_data(save=True):
    mlb = get_label_encoder()
    X, y, startify = get_X_and_encoded_y(encoder=mlb)
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split_stratified(X, y, startify)

    if save:
        os.makedirs(config.data_processed_dev, exist_ok=True)
        np.save(os.path.join(config.data_processed_dev, "X_train.npy"), X_train)
        np.save(os.path.join(config.data_processed_dev, "y_train.npy"), y_train)
        np.save(os.path.join(config.data_processed_dev, "X_test.npy"), X_test)
        np.save(os.path.join(config.data_processed_dev, "y_test.npy"), y_test)
        np.save(os.path.join(config.data_processed_dev, "X_val.npy"), X_val)
        np.save(os.path.join(config.data_processed_dev, "y_val.npy"), y_val)

def load_dev_data():
    X_train = np.load(os.path.join(config.data_processed_dev, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(config.data_processed_dev, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(config.data_processed_dev, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(config.data_processed_dev, "y_test.npy"), allow_pickle=True)
    X_val = np.load(os.path.join(config.data_processed_dev, "X_val.npy"), allow_pickle=True)
    y_val = np.load(os.path.join(config.data_processed_dev, "y_val.npy"), allow_pickle=True)
    return X_train, X_test, X_val, y_train, y_test, y_val
    
def get_label_encoder():
    labels_json_path = list(pathlib.Path(config.data_raw_dev).rglob("*.json"))[0]
    with open(labels_json_path, "r") as f:
        data = json.load(f)
    
    question_answers = {}
    for image in data:
        for label in image["Labels"]:
            if label["Question"] in QUESTIONS_TO_BE_ANSWERED:
                if label["Question"] not in question_answers:
                    question_answers[label["Question"]] = []
                
                # Add answers to the question , then make sure it wasn't added multiple times
                question_answers[label["Question"]] +=  [label["Question"] + "_" + answer for answer in label["Answer"]]  
                question_answers[label["Question"]] = list(set(question_answers[label["Question"]]))
    
    labels = []
    answers_all_questions = list(question_answers.values())
    for q in answers_all_questions:
        labels.extend(q)
    
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    return mlb


def encode_answer(answers, encoder):
    answer_binarized = encoder.transform([answers])
    return answer_binarized[0]

def get_X_and_encoded_y(encoder):
    """
    Reads the json from dev with the images and their answers and builds arrays out of it, for training
    X: (image_id, question)
    y: binary vector of all answers to the questions. Supports multiple answers to a question
    stratify: only needed to make the train test val split stratified by type of abnormality
    """
    labels_json_path = list(pathlib.Path(config.data_raw_dev).rglob("*.json"))[0]
    with open(labels_json_path, "r") as f:
        data = json.load(f)

    no_answer_vector = np.zeros(shape=(117,), dtype=np.int32)
    stratify_question = "Are there any abnormalities in the image?"

    X = []
    y = []
    stratify = [] # feature for stratification
    
    for image in data:
        image_id = image["ImageID"]
        questions = [label["Question"] for label in image["Labels"]]
        answers_all_questions = [label["Answer"] for label in image["Labels"]]
        
        stratify_label = answers_all_questions[questions.index(stratify_question)]
        stratify_label = "_".join(stratify_label)

        for q in QUESTIONS_TO_BE_ANSWERED:
            if q in questions:
                q_index = questions.index(q)
                answers = answers_all_questions[q_index]
                answer_vector = encode_answer([q + "_" + answer for answer in answers] , encoder=encoder)
            else:
                answer_vector = no_answer_vector
                            
            X.append([image_id, q])
            y.append(answer_vector)
            stratify.append(stratify_label)
            
    return X, y, stratify
        
def train_test_val_split_stratified(X, y, stratify):
    X_train, X_test, y_train, y_test, stratify_train, stratify_test = train_test_split(
        X, y, stratify, test_size=0.2, random_state=42, stratify=stratify
    )
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=stratify_test
    )
    return X_train, X_test, X_val, y_train, y_test, y_val




if __name__ == "__main__":
    main()
# end main