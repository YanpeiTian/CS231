import torch
from torch.utils import data
import random

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

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
