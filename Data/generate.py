import os
import numpy as np
import nibabel as nib
import pandas as pd
import csv
import json
import torch
from nilearn import plotting

SOURCEFOLDER = 'img_64_longitudinal/'
TARGETFOLDER = 'training_data/'

def preprocess(data):

    data[data < 1e-8] = 0
    data[data != data] = 0
    data[data == float('Inf')] = 0

    mean = torch.mean(data)
    std = torch.std(data)
    data = (data - mean) / (std + 1e-8)

    return data

def main():
    with open('preprocessed.json') as json_file:
        datapoints = json.load(json_file)


    # Generate .pt file for pytorch training
    # Also do some preprocessing:
    # Eliminate negative value and normalize the data set.
    for datapoint in datapoints:
        img = nib.load(os.path.join(SOURCEFOLDER,datapoint['id']+".nii.gz"))
        img_data = img.get_fdata()
        img_data = torch.Tensor(img_data)
        img_data = preprocess(img_data)

        torch.save(img_data, os.path.join(TARGETFOLDER,datapoint['id']+".pt"))


if __name__ == '__main__':
    main()
