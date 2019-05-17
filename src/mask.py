import torch
from torch import nn
from torch.nn import functional as F

class Mask(nn.Module):
    def __init__(self, model_stream, module):
        super(Mask, self).__init__()
        self.model_stream = model_stream
        self.module = module

    def forward(self, weight, feature):
        result = []
        for i in range(self.model_stream):
            temp_result = self.CAM(weight[i], feature[i])
            result.append(temp_result)
        for i in range(1, self.model_stream):
            for j in range(i):
                if j == 0:
                    mask = result[j]
                else:
                    mask *= result[j]
            mask = torch.cat([mask.unsqueeze(2)] * 4, dim=2)
            self.module.mask_stream[i].data = mask.view(-1).detach()
    
    def CAM(self, weight, feature):
        N, C = weight.shape
        weight = weight.view(N, C, 1, 1, 1).expand_as(feature)
        result = (weight * feature).sum(dim=1)
        result = result.mean(dim=0)

        T, V, M = result.shape
        result = result.view(-1)
        result = 1 - F.softmax(result, dim=0)
        result = F.threshold(result, 0.1, 0)
        result = result.view(T, V, M)
        return result

