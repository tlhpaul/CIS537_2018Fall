# CIS537_2018Fall
CIS 537 Biomedical Image Analysis - Gastounioti/Kontos Project


## Deep learning model
### Shell script
Make sure you have installed conda. Then you can simply run 
```
./cnn.sh
```
It will run from data processing till running the model. If you want to do step by step, please see the follwoing instructions.

### Data processing
Run
```
python rename.py
```

```
python image_processing_7_featmaps.py
```
```
python datasets_adpt_7_featmaps.py
```

#### Set up conda environment
We use conda yaml file to configure the enviornment. You can find
the shell script here: https://conda.io/miniconda.html

#### Install
After running shell script, do the following

anaconda 2 :
```
export PATH=~/anaconda2/bin:$PATH
```
anaconda 3 :
```
export PATH=~/anaconda3/bin:$PATH
```
Then do the following to create and activate the environment
```
conda env create -f environment.yml
```
```
source activate cis537
```
Then when running python files, they will be in this envrionment.

Note, to exit the environment, simply run
```
conda deactivate
```

#### Run the model
Simply run 
```
python CIS537_tensorflow.py
```