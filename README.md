# medVQA-imageClef-Polyps


# Guidelines
Run files as python module, then imports will just work
-m flag and folders separated by dots not backslashes

ex. python -m model.vqamodel

For debugging you will need to start it as a module too.
for that, add the file to the launch.json
when debugging select: debug using launch.json configuration



# Paths
Paths that are used in multiple locations should be defined in config.py
Always build paths using os.path.join pr with Pathlib! 


# Variables
I recommend using_this_type_but_dont_care_too_much_what_you_do

# dependencies
Use
environment_server.yml
environment_win.yml


Export
conda env export > environment.yml

Create
conda env create -f environment.yml

Update
