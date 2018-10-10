# CIS537_2018Fall
CIS 537 Biomedical Image Analysis - Gastounioti/Kontos Project

## Set up conda environment 
We use conda yaml file to configure the enviornment. You can find
the shell file here: https://conda.io/miniconda.html

### Install
After running shell file, do the following

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