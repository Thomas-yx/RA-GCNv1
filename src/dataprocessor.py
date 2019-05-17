import torch
import numpy as np


class Data_transform():
    def __init__(self, data_transform):
        self.data_transform = data_transform

    def __call__(self, x):
        if self.data_transform:
            C, T, V, M = x.shape
            x_new = np.zeros((C*3, T, V, M))
            x_new[:C,:,:,:] = x
            for i in range(T-1):
                x_new[C:(2*C),i,:,:] = x[:,i+1,:,:] - x[:,i,:,:]
            for i in range(V):
                x_new[(2*C):,:,i,:] = x[:,:,i,:] - x[:,:,1,:]
            return x_new
        else:
            return x


class Occlusion_part():
    def __init__(self, occlusion_part=[]):
        self.occlusion_part = occlusion_part

        self.parts = dict()
        self.parts[0] = torch.Tensor([5, 6, 7, 8, 22, 23]).long() - 1              # left arm
        self.parts[1] = torch.Tensor([9, 10, 11, 12, 24, 25]).long() - 1           # right arm
        self.parts[2] = torch.Tensor([22, 23, 24, 25]).long() - 1                  # two hands
        self.parts[3] = torch.Tensor([13, 14, 15, 16, 17, 18, 19, 20]).long() - 1  # two legs
        self.parts[4] = torch.Tensor([1, 2, 3, 4, 21]).long() - 1                  # trunk

    def __call__(self, x):
        for part in self.occlusion_part:
            x[:,:,self.parts[part],:] = 0
        return x


class Occlusion_time():
    def __init__(self, occlusion_time):
        self.occlusion_time = occlusion_time // 2

    def __call__(self, x):
        if not self.occlusion_time == 0:
            x[:,(50-self.occlusion_time):(50+self.occlusion_time),:,:] = 0
        return x

