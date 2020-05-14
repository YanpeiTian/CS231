import os
import numpy as np
import nibabel as nib
import pandas as pd
import csv
import json
from nilearn import plotting

FOLDERNAME = 'img_64_longitudinal/'
LABELFILE = 'ADNI1AND2.csv'

def read_rawdata():
    print("==============================")
    print("Reading raw data from zip files.")
    # Dict to store data associated for each subject.
    subjects = {}

    for filename in sorted(os.listdir(FOLDERNAME)):
        if filename.endswith('gz'):
            subject = filename[:10]
            if subject not in subjects:
                subjects[subject] = []
            img = nib.load(os.path.join(FOLDERNAME,filename))
            try:
                img_data = img.get_fdata()
                subjects[subject].append((filename, img_data))
            except:
                print('Not able to read data from the file: '+filename)

    print("Data extraction from zip file finished.")
    print("==============================")

    return subjects


# Data preprocessing:
# ID, age, label, image
def main():
    # Read raw data from zip files
    subjects = read_rawdata()

    # Label and age information
    labels = pd.read_csv(LABELFILE)

    dataset = []
    print("==============================")
    print("Matching data with labels")
    for subject in subjects:
        datapoints = subjects[subject]
        initial_date = datapoints[0][0][11:21]
        for (filename, data) in datapoints:
            # Get label and age for the current subject.
            try:
                subject_label = labels.loc[labels['Subject_ID']==subject]
                age = min(subject_label['Age'])
                result = min(subject_label['DX_Group'])

                datapoint = {}
                datapoint['id'] = filename[:-7]
                datapoint['subject'] = subject
                datapoint['result'] = result
                # datapoint['data'] = list(data.reshape((64*64*64)))

                date = filename[11:21]
                datapoint['age'] = age + float(date[:4]) - float(initial_date[:4])\
                                       + (float(date[5:7]) - float(initial_date[5:7]))/12\
                                       + (float(date[8:]) - float(initial_date[8:]))/360

                dataset.append(datapoint)

            except:
                print("Cannot find "+subject+" in the label file.")

    print("Matching data with label finished.")
    print("==============================")

    with open('preprocessed.json', 'w') as fp:
        json.dump(dataset, fp)

    # print(len(dataset[0]['data']))


# Load a datapoint and visualize
def load():
    with open('preprocessed_small.json') as json_file:
        data = json.load(json_file)

    img = data[0]['data']
    img = np.array(img).reshape((64,64,64))
    print(img.shape)

    new_image = nib.Nifti1Image(img, affine=np.eye(4))
    plotting.plot_img(new_image)
    plotting.show()

    print(len(data))


if __name__ == '__main__':
    main()
    # load()
