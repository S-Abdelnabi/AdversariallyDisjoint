import torch
import torch.nn as nn
import torch.nn.functional as F

class MyEnsemble(nn.Module):
    def __init__(self, models_list):
        super(MyEnsemble, self).__init__()
        self.models_list = models_list

    def forward(self, x):
        output_list = []
        for i in range(0,len(self.models_list)):
            out = self.models_list[i](x) 
            output_list.append(out)
        average_out = output_list[0]
        for i in range(1,len(self.models_list)):
            average_out = average_out + output_list[i] 
        average_out = average_out / len(self.models_list)
        return average_out

    def forward_subset(self, x, idx_list):
        output_list = []
        for idx in idx_list:
            out = self.models_list[idx](x) 
            output_list.append(out)
        average_out = output_list[0]
        for i in range(1,len(idx_list)):
            average_out = average_out + output_list[i] 
        average_out = average_out / len(idx_list)
        return average_out