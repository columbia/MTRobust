import torch.nn as nn
import math
import torch.nn.functional as F

class Ensemble(nn.Module):

    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.model_list = []
        for each in model_list:
            self.model_list.append(each)

    def forward(self, x):
        output = []
        for each in self.model_list:
            output.append(each(x))
        return output

    def eval(self):
        for i, each in enumerate(self.model_list):
            self.model_list[i] = each.eval()

        # this eval is needed for eval!