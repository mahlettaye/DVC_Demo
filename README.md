### Demo On how to set DVC with MLFlow on local repository 
The Project is implemented on using Data version control tool.
# To set Up DVC 
Create folder DVC with in that Create folder data.
Put your raw data into data folder 
Using Git /Git bash  
# git init   # to initialize the git 
# dvc init    # to Initialize DVC 

# dvc remote add -d dvc-remote /temp/dvc-storage  # to create remote storage on your local machine

# dvc add data/wine-quality.csv    # to track data set on dvc 

# Git add data/.gitignore data/wine-quality.csv.dvc
This two files are created by dvc when you apply add function 
The .gitignore store index information about your data
The wine-quality.csv.dvc stores 

# git commit -m 'initial data is added'
# git tag -a 'v1' -m 'raw data'
# dvc push # to push changes into dvc



