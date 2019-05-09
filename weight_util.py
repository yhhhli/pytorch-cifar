import torch
import torch.nn as nn
import numpy as np


class TernarizeOp:
    def __init__(self, model, weight_norm=False):
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(1, count_targets - 2, count_targets - 2).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        index = -1
        if weight_norm:
            index = -1
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    index += 1
                    if index in self.ternarize_range:
                        tmp = m.weight_v.data.clone()
                        self.saved_params.append(tmp)  # tensor
                        self.target_modules.append(m.weight_v)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    index += 1
                    if index in self.ternarize_range:
                        tmp = m.weight.data.clone()
                        self.saved_params.append(tmp)  # tensor
                        self.target_modules.append(m.weight)  # Paramete


    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True). \
                mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-1.0, 1.0)

    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            tensor = self.target_modules[index].data
            delta = self.Delta(tensor)
            #s = tensor.size()
            position = tensor.abs().gt(delta).type(torch.cuda.FloatTensor)
            #alpha = position.mul(tensor)
            #n = position.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True)
            #alpha = alpha.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            #self.target_modules[index].data = self.target_modules[index].data.sign().mul(alpha.mul(position))
            self.target_modules[index].data = self.target_modules[index].data.sign().mul(position)

    def Delta(self, tensor):
        n = tensor[0].nelement()
        if (len(tensor.size()) == 4):  # convolution layer
            delta = 0.7 * tensor.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
        elif (len(tensor.size()) == 2):  # fc layer
            delta = 0.7 * tensor.norm(1, 1, keepdim=True).div(n)
        return delta

    def Ternarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.SaveWeights()
        self.TernarizeWeights()

    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
