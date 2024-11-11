import os
import matplotlib.pyplot as plt
import cv2
import config
from utils import preprocess_image
from tqdm import tqdm


def main():
    dataPath = os.path.join(config.data_path, "images")
    for _, _, filenames in os.walk(dataPath):
        for filename in tqdm(filenames):
            imagepath = os.path.join(dataPath, filename)
            processed = getProcessed(imagepath)
            saveProcessed(processed, filename)

def saveProcessed(processed, filename):
    saveName = os.path.join(config.data_processed, filename)
    cv2.imwrite(saveName, processed)

def getProcessed(imagepath):
    img = cv2.imread(imagepath)
    processed, inpainted = preprocess_image(img)
    return processed

def showImage(imagepath):
    img = cv2.imread(imagepath)
    processed, inpainted = preprocess_image(img)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title("Raw image")
    axes[0].axis('off')

    axes[1].imshow(inpainted)
    axes[1].set_title("inpainted")
    axes[1].axis('off')

    axes[2].imshow(processed)
    axes[2].set_title("blackmask removal")
    axes[2].axis('off')

    plt.show()

if __name__ == "__main__":
    main()