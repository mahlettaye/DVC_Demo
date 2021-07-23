### Demo On how to set DVC with MLFlow on local repository 
The Project is implemented on using Data version control tool.
### To set Up DVC 
-Create folder DVC with in that Create folder data.
-Put your raw data into data folder 
-Using Git /Git bash  
-`git init    to initialize the git` 
-` dvc init     to Initialize DVC` 

-`dvc remote add -d dvc-remote /temp/dvc-storage`  # to create remote storage on your local machine

-`dvc add data/wine-quality.csv`    # to track data set on dvc 

-` Git add data/.gitignore data/wine-quality.csv.dvc`
This two files are created by dvc when you apply add function 
The .gitignore store name of data file and it will be used to not push data to github 
The wine-quality.csv.dvc stores information about file with hashed value and data path 

-`git commit -m 'initial data is added'`
-` git tag -a 'v1' -m 'raw data'` TO identify the exact version of your dataset 

-`dvc push` # to push changes into dvc remote repository
- To subset your data 
-`set -i '2,1001d" data/wine-quality.csv`
- `dvc add data/wine-quality.csv`  to add new subset data 
- The same step as previous will repeated with V2 tag
- `dvc Push`  to push change 
#### train .py contains  simple model for wine quality prediction.




