# TO-DO

- Prepare data to train model
    - take the 18 questions from https://www.imageclef.org/2023/medical/vqa task 1
    - Bring the Labels into a format that can be used to train the model
        - Extract all answers
        - Multi label binarizer
        - each of the possible answers are a node in the output layer (sigmoid)
        - Giving the answer o/1 is based on a 0.5 threshold
        - Loss function is binary crossentropy / sparse categorical crossentropy

    - Input for model is:
        - Image
        - Binary Vector, encoding all the possible answers
    
        - Model predicts that vector -> Binary Crossentropy loss

    - Encoder the answers to a vector before training
        