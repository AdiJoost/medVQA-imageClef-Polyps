# medVQA-imageClef-Polyps
## Data
Get the Data from:
    devset_url = https://drive.google.com/uc?id=1jTyLWwcHzbLpWjSNwmgiiavXDjuQe5y7&export=download
    testset_url= https://drive.google.com/uc?id=1PQPiOkyfQrLJ5wWxkHZy_FYIdL9hXiMl&export=download

Unzip them into the data/raw/dev and data/raw/test folders

## Guidelines
Run files as python module, then imports will just work
-m flag and folders separated by dots not backslashes

ex. python -m model.vqamodel

For debugging you will need to start it as a module too.
for that, add the file to the launch.json
when debugging select: debug using launch.json configuration



## Paths
Paths that are used in multiple locations should be defined in config.py
Always build paths using os.path.join or with Pathlib! 


## Variables
I recommend using_this_type_but_dont_care_too_much_what_you_do

## dependencies
Use
environment_server.yml
environment_win.yml


Export
conda env export > environment.yml

Create
conda env create -f environment.yml

Update


# gpu

For training / testing use the test.bash / trainer.bash
They will execute the corresponding python files on the gpu



# vision instruct
For the vision instruct model you will need to get a access token , follow this guide https://drlee.io/llama-3-2-a-step-by-step-guide-to-language-vision-and-fine-tuning-in-google-colab-260157fba89f
and https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login