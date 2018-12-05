#!/bin/bash
python rename.py
python image_processing_7_featmaps.py
python datasets_adpt_7_featmaps.py
conda env create -f environment.yml
source activate cis537
python CIS537_tensorflow.py