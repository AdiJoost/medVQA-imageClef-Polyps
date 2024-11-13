import tensorflow as tf
import os
from keras import Model, models,  layers, utils, callbacks, optimizers, losses, metrics
import config
import datetime
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel, TfAutoModelForImageClassification
import tensorflow_hub as hub
import numpy as np


MODEL_NAME = os.path.basename(__file__).split(".")[0] + ".keras"
MODEL_PATH = os.path.join(config.trained_model_path, MODEL_NAME)
MODEL_LOG_DIR = os.path.join(config.train_logs_path, os.path.basename(__file__).split(".")[0])
if not os.path.exists(MODEL_LOG_DIR):
    os.makedirs(MODEL_LOG_DIR)

from dataset_tf import get_dev_datasets
    
    
def main():
    
    train_ds, test_ds, val_ds = get_dev_datasets()
    
    model = VQA()
    
    
    
    """
    Not sure if metrics are correct:
        
    function, which combines a Sigmoid layer and the BCELoss, is applied
    in the training process. After each epoch, the training loss and validation loss are calculated,
    and the performance are then evaluated on classification metrics such as accuracy, precision,
    recall and F1-Score. To ensure a meaningful result for multi-label classification, the metrics are
    calculated using ground truth and prediction sets of binary vectors, in which recall, precision
    and F1-scores should be calculated on each sample and find their average. The model’s state
    that obtains best F1-Score is used for prediction in the testing phase.
    """
    model.compile(optimizer=get_optimizer(),
                  loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics={
                      "accuracy" : metrics.Accuracy(),
                      "precision" : metrics.Precision(),
                      "recall": metrics.Recall(),
                      "f1": metrics.F1Score()
                  }
                  )
    
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=15,
                        batch_size=64,
                        callbacks=get_callbacks())





@utils.register_keras_serializable()
class VQA(Model):
    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder()
        self.classifier = Classifier()
        

    def call(self, input):
        image_embedding = self.image_encoder(input[0])
        question_embedding = self.question_encoder(input[1])
        probabilities = self.classifier((question_embedding, image_embedding))
        return probabilities


@utils.register_keras_serializable()
class ImageEncoder(layers.Layer):   
    """
    Takes the image as input and returns an embedding vector of size 512
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        base_model = AutoModelForImageClassification.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")  
        
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
    """
    Takes the question as input and returns an embedding vector
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.tokenizer = AutoModel.from_pretrained('bert-base-uncased')
        
    def call(self, question):
         #if isinstance(question, bytes)  else question # tf whyyyyyyyy??? you do this to me why is string now byte f you
        outputs = self.bert_model.signatures['tokens'](question)
        question_embedding = outputs['pooled_output']
        return question_embedding

@utils.register_keras_serializable()
class Classifier(layers.Layer):   
    """
    Takes the two embeddings, concatenates them and then predicts the labels
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.num_labels = 117 
                
        self.dense_fc = layers.Dense(units=None, activation="relu")
        self.dropout = layers.Dropout(rate=0.5)
        self.dense_output = layers.Dense(self.num_labels, activation='sigmoid')

        
    def call(self, qEmbedding_imgEmbedding):
        q_embedding, img_embedding = qEmbedding_imgEmbedding
        
        full_embedding = np.concatenate([q_embedding, img_embedding])
        self.dense_fc.units = full_embedding.shape[0]
        x = self.dense_fc(full_embedding)
        x = self.dropout(x)
        probabilities = self.dense_output(x)
        return probabilities
        
        
        
        
    
    

def get_callbacks():
    """
    Returns a usefull collection of callbacks
    ModelCheckpoint, Tensorboard
    Also adds a linear decaying learning rate
    Creates log dir if it does not exist 
    Returns:
        List[callbacks]: A list of keras callbacks
    """
    checkpoint = callbacks.ModelCheckpoint( 
        filepath=MODEL_PATH,
        save_best_only=True, 
        save_weights_only=False,
        mode="min",
        monitor="rec"
    )
    
    def linear_lr_scheduler(epoch, lr):
        decay_rate = 0.9333  # 100% - 6.67%
        return lr * decay_rate
    lr_schedule = callbacks.LearningRateScheduler(schedule=linear_lr_scheduler)
    
    log_dir = os.path.join(config.train_logs_path, MODEL_NAME.split(".")[0], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir,exist_ok=True)
    tensorboard = callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)    
    
    return [checkpoint, tensorboard, lr_schedule]


def get_optimizer():
    """
    Returns the optimizer Adam with weighted decay 
    Linear learning rate decay is implemented via callback
    
    Returns:
        keras.optimizer: AdamW optimizer 
    """
    
    return optimizers.AdamW(learning_rate=5e-5)




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