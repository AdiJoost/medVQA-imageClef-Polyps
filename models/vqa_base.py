import tensorflow as tf
import os
from keras import Model, models,  layers, utils, callbacks, optimizers
from keras.applications import ResNet50

import config
import datetime
from transformers import ViTFeatureExtractor, TFViTModel,BertTokenizer
import tensorflow_hub as hub




MODEL_NAME = os.path.basename(__file__).split(".")[0] + ".keras"
MODEL_PATH = os.path.join(config.trained_model_path, MODEL_NAME)
MODEL_LOG_DIR = os.path.join(config.train_logs_path, __file__)
if not os.path.exists(MODEL_LOG_DIR):
    os.makedirs(MODEL_LOG_DIR)

    
    
def main():
    print(1+1)



@utils.register_keras_serializable()
class VQA(Model):
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.num_labels = 10 # dont know what that will be
        
        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder()
        self.dense = layers.Dense(256, activation="relu")
        self.dense_output = layers.Dense(self.num_labels, activation='sigmoid')

    def call(self, input):
        image_embedding = self.image_encoder(input[0])
        question_embedding = self.question_encoder(input[1])
        x = layers.Concatenate()([image_embedding, question_embedding])
        x = self.dense(x)
        output = self.dense_output(x)
        return output
        


@utils.register_keras_serializable()
class ImageEncoder(layers.Layer):   
    def __init__(self,**kwargs):
        super.__init__(**kwargs)

        base_model = ResNet50(weights='imagenet', include_top=False)
        
        
        self.pre_trained = base_model
        self.pooling = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(512, activation="relu")

    def call(self, image):
        x = self.pre_trained(image)
        x = self.pooling(x)
        image_embedding = self.dense(x)
        return image_embedding



@utils.register_keras_serializable()
class QuestionEncoder(layers.Layer):   
    def __init__(self,**kwargs):
        super.__init__(**kwargs)
        
        self.bert_model = hub.load("https://tfhub.dev/google/experts/bert/wiki_books/sst2/2")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def call(self, question):
        inputs = self.tokenizer(question, return_tensors='tf', padding='max_length', max_length=64, truncation=True)
        outputs = self.bert_model.signatures['tokens'](inputs)
        question_embedding = outputs['pooled_output']
        return question_embedding

# callbacks and optimizer need to be adjusted
def getCallbacks():
    early_stop = callbacks.EarlyStopping(monitor="rec", mode="min", patience=120, min_delta=0.0002)
    checkpoint = callbacks.ModelCheckpoint( 
        filepath=MODEL_PATH,
        save_best_only=True, 
        save_weights_only=False,
        mode="min",
        monitor="rec"
    )
    log_dir = os.path.join(config.train_logs_path, MODEL_NAME.split(".")[0], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir,exist_ok=True)
    tensorboard = callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)    
    
    return [early_stop, checkpoint, tensorboard]

def getOptimizer():
    initial_learning_rate = 0.01
    # lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate,
    #                                 decay_steps=2000,
    #                                 decay_rate=0.5)
    return optimizers.Adam(learning_rate=initial_learning_rate)



def load_model(self):
    custom_Objects = {
        "VQA": VQA,
        "ImageEncoder": ImageEncoder,
        "QuestionEncoder": QuestionEncoder    
    }
    return models.load_model(self.model_save_path, custom_objects=custom_Objects) # , compile=False
    


if __name__ == "__main__":
    main()
# end main