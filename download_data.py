import pandas as pd
import opendatasets as od

# INSTRUCTIONS to download:
# 1. Copy this file into your $PSCRATCH directory
# 2. Create a file kaggle.json in $PSCRATCH with content:
#   {"username":"XXXX","key":"XXXXX..."}
# 3. module load python (maybe you need to install opendatasets, idr)
# 4. python3 download_data.py

dataset = 'https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data'
od.download(dataset)
