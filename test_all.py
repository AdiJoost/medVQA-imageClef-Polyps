

import test
# MODEL_NAME = "vqa_model_dinov2.pth"
# model_id = "facebook/dinov2-large"

test_list = [
    # ("vqa_model_dinov2.pth", "facebook/dinov2-large"), 
    ("vqa_aimv2.pth", "apple/aimv2-large-patch14-224")
]



counter = 1
for entry in test_list:
    model_name = entry[0]
    model_id = entry[1]
    print("start test")
    test.run_test(model_name=model_name, model_id=model_id)
    
    print(f"[{counter}/{test_list.__len__()}] finished test of {model_name}, {model_id}")
    print(f"find the results at results_dev_test_{model_name.split(".")[0]}.txt")
    
    
print("Performance Test finished")