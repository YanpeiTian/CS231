import torch.nn as nn
import torch.nn.functional as F


class Test_Classifier(nn.Module):
    def __init__(self):
        super(Test_Classifier, self).__init__()
        self.classifier = nn.Linear(64*64*64, 2)

    def forward(self, x):
        x = x.reshape((-1, 64*64*64))
        x = self.classifier(x)
        return x
