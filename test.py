import pandas as pd
import numpy as np
import random
import time
from Compress import compress_zstd, compress_lz4, compress_gzip
from Decompress import decompress_zstd, decompress_lz4, decompress_gzip
from transformers import pipeline  
from transformers import pipeline, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

original = pd.read_csv('../data/train_40k.csv')
mask_gzip = pd.read_csv('../Output/mask_gzip.csv')
mask_lz4 = pd.read_csv('../Output/mask_lz4.csv')
mask_zstd = pd.read_csv('../Output/mask_zstd.csv')

mask_gzip['diff'] = mask_gzip['Predict_Text'] != original['Text']
mask_lz4['diff'] = mask_lz4['Predict_Text'] != original['Text']
mask_zstd['diff'] = mask_zstd['Predict_Text'] != original['Text']

print('Masked Texts with gzip:', mask_gzip['diff'].value_counts())
print('Masked Texts with lz4:', mask_lz4['diff'].value_counts())
print('Masked Texts with zstd:', mask_zstd['diff'].value_counts())