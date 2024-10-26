
import config
import pathlib
import json



labels_json_path = list(pathlib.Path(config.data_raw_dev).rglob("*.json"))[0]


with open(labels_json_path, "r") as f:
    data = json.load(f)


# ## Not all of the questions are for Task 1
# 
# for example
# 
#   {'Question': 'Where exactly in the image is the instrument located?',
#    'AnswerType': 'segmentation',
#    'Answer': 'clb0lbwzadoyc086u0brshvx5_mask.png'}
# 
#    is not 


questions_to_be_answered = ["What type of procedure is the image taken from?",
"How many instrumnets are in the image?",
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
"What type of polyp is present?"]


question_answers = {}
for image in data:
    for label in image["Labels"]:
        if label["Question"] in questions_to_be_answered:
            if label["Question"] not in question_answers:
                question_answers[label["Question"]] = []
            
            # Add answers to the question , then make sure it wasn't added multiple times
            question_answers[label["Question"]] +=  [label["Question"] + "_" + answer for answer in label["Answer"]]  
            question_answers[label["Question"]] = list(set(question_answers[label["Question"]]))


# Answers seem good, besides that there are two colors in "How many findings are present?"


# for image in data:
#     for label in image["Labels"]:
#         if label["Question"] in questions_to_be_answered:
#             if label["Question"] == "How many findings are present?":
#                 print(image["ImageID"], label["Answer"])


# There are some bad labels in there...
# How many is not a question you can answer with pink, yellow 
# But we'll just leave it in for now, the model should never give it as an answer. probably


labels = []
answers = list(question_answers.values())
for q in answers:
    labels.extend(q)
labels


# # Encoding


# For example if an image has the question - answer
#  'Are there any instruments in the image?_Polyp snare',
#  'Are there any instruments in the image?_Biopsy forceps'
# 
# There should be a vector with those two values = 1


sample_labels = [
    ['What color is the abnormality?_Yellow', 'What is the size of the polyp?_< 5mm'],
    ['What color is the anatomical landmark?_Red', 'What is the size of the polyp?_11-20mm'],
    ['What color is the abnormality?_grey', 'What color is the anatomical landmark?_White']
]


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([labels])
mlb.classes_[0:10]


encoded_samples  = mlb.transform(sample_labels)
encoded_samples 


answer = mlb.inverse_transform(encoded_samples)
answer


