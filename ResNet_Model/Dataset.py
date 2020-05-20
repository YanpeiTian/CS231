import torch
from torch.utils import data
import numpy as np
import random
import scipy.ndimage

# import nibabel as nib
# from nilearn import plotting

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, aug_factor=0, aug_strength=1):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.aug_factor = aug_factor
        self.aug_strength = aug_strength

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('../Data/training_data/' + ID + '.pt')

        y = self.labels[ID]

        # Data augmentation for training set
        new_datas = [X]
        if (self.aug_factor > 0):
            for i in range(0,self.aug_factor):
                new_data = X.clone().numpy()

                new_data = scipy.ndimage.interpolation.rotate(new_data, np.random.uniform(-self.aug_strength,self.aug_strength), axes=(1,0), reshape=False)
                new_data = scipy.ndimage.interpolation.rotate(new_data, np.random.uniform(-self.aug_strength,self.aug_strength), axes=(0,2), reshape=False)
                new_data = scipy.ndimage.interpolation.rotate(new_data, np.random.uniform(-self.aug_strength,self.aug_strength), axes=(1,2), reshape=False)
                new_data = scipy.ndimage.shift(new_data, np.random.uniform(-self.aug_strength,self.aug_strength))

                new_data = torch.tensor(new_data)
                new_datas.append(new_data)

            X = torch.stack(new_datas)
            ys = torch.zeros(self.aug_factor+1)
            ys.fill_(y)
            y = ys.long()

        return X, y

def generate(datapoints):
    IDs = []
    labels = {}
    for datapoint in datapoints:
        IDs.append(datapoint['id'])
        if (datapoint['result'] == 'Normal'):
            labels[datapoint['id']] = 0
        else:
            labels[datapoint['id']] = 1

    return IDs, labels


def train_valid_test_split(datapoints, split_ratio=(0.8, 0.1, 0.1)):
    """
    split the datapoints list into 3 parts based on the subjects
    :param datapoints: list of image info dictionary
    :param split_ratio: (train, validation, test) ratio
    :return: 3 lists of datapoints for train, validation, and test
    """

    # get the total number of subjects in the whole datapoints:

    all_subjects = set()
    for datapoint in datapoints:
        all_subjects.add(datapoint['subject']) # put all subject code into a set

    num_subjects = len(all_subjects)

    num_train = int(split_ratio[0]*num_subjects) # in terms of subjects, not images
    num_val = int(split_ratio[1]*num_subjects)

    all_subjects = list(all_subjects)
    random.shuffle(all_subjects)

    train_set = set(all_subjects[:num_train])
    val_set = set(all_subjects[num_train:num_train+num_val])
    test_set = set(all_subjects[num_train+num_val:])

    train_list, val_list, test_list = [], [], []

    for index, datapoint in enumerate(datapoints):
        if datapoint['subject'] in train_set:
            train_list.append(datapoint)
        elif datapoint['subject'] in val_set:
            val_list.append(datapoint)
        elif datapoint['subject'] in test_set:
            test_list.append(datapoint)

    return (train_list, val_list, test_list)

def balanced_data_split(datapoints, split_ratio=(0.8, 0.2)):
    """
    split the datapoints list into 2 parts based on the subjects
    :param datapoints: list of image info dictionary
    :param split_ratio: (train, validation) ratio
    :return: 2 lists of datapoints for train, validation
    :
    Only keep Normal and AD datapoints, make a balanced training set.
    """
    Normal_subjects = set()
    AD_subjects = set()
    for datapoint in datapoints:
        if (datapoint['result']=='AD'):
            AD_subjects.add(datapoint['subject'])
        if (datapoint['result']=='Normal'):
            Normal_subjects.add(datapoint['subject'])

    Normal_subjects = list(Normal_subjects)
    AD_subjects = list(AD_subjects)

    train_size = int(split_ratio[0] * (len(Normal_subjects) + len(AD_subjects))) // 2

    if (train_size > len(Normal_subjects) or train_size > len(Normal_subjects)):
        print("Cannot make balanced training set. Please decrease the percentage of training.")

    train_set = Normal_subjects[:train_size] + AD_subjects[:train_size]
    val_set = Normal_subjects[train_size:] + AD_subjects[train_size:]

    random.shuffle(train_set)
    random.shuffle(val_set)

    train_set = set(train_set)
    val_set = set(val_set)

    train_list, val_list = [], []

    for index, datapoint in enumerate(datapoints):
        if datapoint['subject'] in train_set:
            train_list.append(datapoint)
        elif datapoint['subject'] in val_set:
            val_list.append(datapoint)

    return train_list, val_list
