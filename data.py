import torch
from torch.utils.data import Dataset
import random
from util import *

beta = np.pi/3

class MyDataset(Dataset):
    def func(self, x, y, beta):
        return fitRotateFunc(x, y, beta=beta)
    
    def __len__(self):
        return self.len

    def __init__(self, width = 10, height = 10, beta = beta):
        super().__init__()
        self.len = width*height
        self.beta = beta
        len = 2*np.pi
        self.x, self.y = getGrid(-len, len, width, -len, len, height)
        self.z = self.func(self.x, self.y, beta=beta)
        
    def __getitem__(self, index):
        xy = torch.tensor((self.x[index], self.y[index]), dtype=torch.float32)
        z = torch.tensor([self.z[index]], dtype=torch.float32)
        return xy, z

def find_period(array):
    n = len(array)
    
    # 遍历数组的长度范围，从1到数组长度的一半
    for length in range(1, n // 2 + 1):
        if n % length == 0:
            is_period = True
            
            # 检查数组是否由重复子数组构成
            for i in range(length, n):
                if array[i] != array[i % length]:
                    is_period = False
                    break
            
            # 如果是周期数组，则返回周期值
            if is_period:
                return length
    
    # 如果没有找到周期，则返回-1表示不是周期数组
    return -1

def getContentTensor(beta = beta):
    angles = (2*np.pi/360) * np.arange(360)
    radii = 1.0
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return torch.from_numpy(fitRotateFunc(x, y, beta=beta))

if __name__ == '__main__':
    data = MyDataset()
    print(data[35][0])
    print(data[35][1])
    print(getContentTensor().shape)