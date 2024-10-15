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

# Download data to data/raw folder 

# then apply image preprocessing and store images in data/processed folder

# define method to load data to use for training / testing / validation

# right now just code that will be somewhat simmilar (from another project)


def main():
    get_data()


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

    



    

def prepare_all_wafer_images():
    ok_raw = [os.path.join(config.wafer_ok_raw_path, filename) for filename in os.listdir(config.wafer_ok_raw_path)]
    nok_raw = [os.path.join(config.wafer_nok_raw_path, filename) for filename in os.listdir(config.wafer_nok_raw_path)]
    
    prepare_wafer_images(ok_raw, config.wafer_ok_processed_path)
    prepare_wafer_images(nok_raw, config.wafer_nok_processed_path)

def prepare_wafer_images(file_list = [], output_dir=""):
    """
    Gets the raw images of the wafer dataset and crops to the central rectangle
    
    processed images are stored in "processed/ok" and "processed/nok"
    """
    for img in file_list:
        prepare_wafer_image(img, output_dir)
    
def prepare_wafer_image(filepath=None, destination_dir=config.data_path):
    """
    Takes raw image of wafer and prepares it for training/inference
    """
    if not os.path.exists(filepath):
        print(f"{filepath} does not exists")
        return

    os.makedirs(destination_dir, exist_ok=True)
    (_ , filename) = os.path.split(filepath)
    destination_path = os.path.join(destination_dir, filename)
    if os.path.exists(destination_path):
        print(f"Image exists already, is overwritten: {destination_path}")
    
    
    img = plt.imread(filepath)    
    thresh_img = apply_threshold(img)
    contour, rect, box_points = get_wafer_contour(thresh_img)
    cropped_rotated = crop_to_rect(img, rect, box_points)
    processed = apply_preprecessing(cropped_rotated)
    
    cv2.imwrite(destination_path, processed)
    
def apply_preprecessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)    
    _, binarized = cv2.threshold(normalized_image, 105,255, cv2.THRESH_BINARY)    
    return binarized
    
def apply_threshold(img, show_img=False):
    """
    Apply Gaussian Blur and otsu threshold
    """
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) 
    blurred_img = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, thresh_img = cv2.threshold(blurred_img,0,100,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if show_img:
        plt.imshow(thresh_img, "gray")
        plt.title("Blurred & Threshold")
        plt.xticks([])
        plt.yticks([])    
        plt.show()
        
    return thresh_img
    
def get_wafer_contour(thresh_img):
    """
    Gets the contour, corner points, rectangle of the wafer
    """
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #[-2] 
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour) # min area rectangle around contour
    box_points = cv2.boxPoints(rect) # edge points 
    box_points = np.intp(box_points)
    return max_contour, rect, box_points

def crop_to_rect(original, rect, box, show_img=False):
    """
    Rotates img to align with axis and crops
    """
    center = rect[0]
    angle = rect[2]
    
    # Rotate
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(original, rotation_matrix, (original.shape[1], original.shape[0]))

    # Crop
    edge_takeoff = 15 # take a bit off the sides to remove border
    x, y, w, h = cv2.boundingRect(box)
    cropped_rotated = rotated_image[y + edge_takeoff:y+h - edge_takeoff, x + edge_takeoff:x+w - edge_takeoff]
    
    if show_img:
        plt.imshow(cropped_rotated)
        plt.title("Cropped & Rotated")
        plt.xticks([])
        plt.yticks([])    
        plt.show()
        
    return cropped_rotated
    


# example load image dataset from directory, adjust as necessary

# def get_wafer_dataset_nok(batch_size=8, image_size=(512,512)):
#     (train_ds, val_ds) = utils.image_dataset_from_directory(
#         config.wafer_nok_processed_path,
#         batch_size=None,
#         subset="both",
#         validation_split=0.1,
#         labels=None,
#         seed=42,
#         color_mode="grayscale",
#         image_size=image_size
#     )
    # def normalize(image):
    #     image = tf.cast(image, tf.float32) / 255.0
    #     return image
#     train_ds = (train_ds.map(lambda x: normalize(x))
#                         .batch(batch_size, drop_remainder=True)
#                         .prefetch(buffer_size=tf.data.AUTOTUNE))
    
#     val_ds = (val_ds.map(lambda x: normalize(x))
#                     .batch(batch_size, drop_remainder=True)
#                     .prefetch(buffer_size=tf.data.AUTOTUNE))
    # return (train_ds, val_ds)    

    



if __name__ == "__main__":
    main()
# end main