import torch
from torch import nn
import torch.nn.functional as F
from data import *
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
使用自注意力机制
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=16, num_heads=4) -> None:
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.q_linear = nn. Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.ln1 = nn.BatchNorm1d(num_features=self.hidden_size)

        self.dropout = nn.Dropout(p=0.5)
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.ln2 = nn.BatchNorm1d(num_features=self.hidden_size)
        
    def forward(self, x): 
        batch_size = x.shape[0]
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_size)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)  # (batch_size, seq_len, hidden_size)
        out = self.ln1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + x
        res = out
        out = self.out_linear(out)  # (batch_size, seq_len, hidden_size)
        out = self.ln2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + res
        return out

'''旋转不变性网络-直接旋转点的位置至0~beta后进行训练/预测'''
class MyRotateNet2(nn.Module):
    def __init__(self, input_size=2, seq_len=64, hidden_size=8, num_heads=4) -> None:
        super(MyRotateNet2, self).__init__()
        self.input_size = input_size
        self.input = nn.Sequential(
            nn.Linear(input_size, seq_len), nn.ReLU(),
            nn.Linear(seq_len, seq_len) 
        )
        self.conv = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1) # 1D convolution layer
        self.multi_head_attention1 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention2 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.multi_head_attention3 = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layer = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 1)
        )

    # 旋转batch内所有的点到0~beta角度内
    def transform(self, data, beta):
        x = data[:, 0]  # (batch_size,)
        y = data[:, 1]  # (batch_size,)
        angles = torch.atan2(y, x)  # 计算原始点的角度
        radii = torch.sqrt(x**2 + y**2)  # 计算原始点的长度
        while torch.any( angles<=2*torch.pi ):
            angles = torch.where(angles<=2*torch.pi, angles+beta, angles)
        
        a = radii * torch.cos(angles)  # 计算旋转后的点的x坐标
        b = radii * torch.sin(angles)  # 计算旋转后的点的y坐标
        combined_array = torch.stack([a, b], dim=1)
        return combined_array

    def forward(self, x, beta):
        x = self.transform(data=x, beta=beta)
        out = self.input(x)
        out = out.unsqueeze(1)  # (batch_size, 1, seq_len)
        out = self.conv(out)  # (batch_size, hidden_size, seq_len)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)
        
        out = self.multi_head_attention1(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention2(out)  # (batch_size, seq_len, hidden_size)
        out = self.multi_head_attention3(out)  # (batch_size, seq_len, hidden_size)
        
        out = out.reshape(out.size(0), -1)  # (batch_size, seq_len * hidden_size)
        out = self.layer(out)  # (batch_size, 1)
        return out

class AngleNet(nn.Module):
    def __init__(self):
        super(AngleNet, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(360, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        output = 2 * torch.pi * x  # 将输出映射到0~2π之间
        return output

class CombinedNet(nn.Module):
    def __init__(self, check=False):
        super(CombinedNet, self).__init__()
        def init_weights_small(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight, -0.001, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.001)  # Set the bias to a small number

        angle_net = AngleNet()
        angle_net.apply(init_weights_small)
        self.angle_net = angle_net
        self.content_tensor = getContentTensor().unsqueeze(0).to(torch.float32).to(device)
        self.rotate_net = MyRotateNet2()
        self.check = check

    def forward(self, x):
        if self.check:
            angle = np.pi/2
        else:
            angle = self.angle_net(self.content_tensor)[0][0]
        pred = self.rotate_net(x, beta=angle)
        return pred, angle

class restrictCombinedNet(nn.Module):
    def __init__(self, check=False):
        super(restrictCombinedNet, self).__init__()

        content_tensor = getContentTensor().unsqueeze(0).to(torch.float32).to(device)
        content_array = content_tensor.squeeze().cpu().numpy()
        period = find_period(content_array)
        self.angle = 2*np.pi*period/360
        self.rotate_net = MyRotateNet2()
        self.check = check

    def forward(self, x):
        if self.check:
            self.angle = np.pi/2
        pred = self.rotate_net(x, beta=self.angle)
        return pred, self.angle

def checkRotateStable():
    x1 = torch.tensor([1,1], dtype=torch.float32).view(1, 2)
    x2 = torch.tensor([-1,1], dtype=torch.float32).view(1, 2)
    x3 = torch.tensor([1,-1], dtype=torch.float32).view(1, 2)
    x4 = torch.tensor([-1,-1], dtype=torch.float32).view(1, 2)
    contentTensor = getContentTensor().unsqueeze(0).to(torch.float32)
    
    net = CombinedNet(check=True).eval()
    pred1, _ = net(x1, contentTensor)
    pred2, _ = net(x2, contentTensor)
    pred3, _ = net(x3, contentTensor)
    pred4, _ = net(x4, contentTensor)
    print("check Output1", pred1)
    print("check Output2", pred2)
    print("check Output3", pred3)
    print("check Output4", pred4)

def checkGraid():
    x = torch.randn(8,360,dtype=torch.float32)
    contentTensor = getContentTensor().unsqueeze(0).to(torch.float32)
    net = CombinedNet()
    pred, _ = net(x, contentTensor)
    target = torch.randn(8, 1)
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    print(loss)
    # 执行反向传播
    net.zero_grad()
    loss.backward()
    # 打印每一层的梯度
    print("梯度信息：")
    for name, param in net.named_parameters():
        if param.grad is not None:
            print(name)

if __name__ == '__main__':
    #checkRotateStable()
    #checkGraid()
    device = 'cpu'
    x = torch.randn(8,360,dtype=torch.float32)
    contentTensor = getContentTensor().unsqueeze(0).to(torch.float32)
    content_array = contentTensor.squeeze().cpu().numpy()
    period = find_period(content_array)
    print(2*np.pi*period/360)
    
    net = CombinedNet()
    pred, angle = net(x, contentTensor)
    print("Input  Shape:", x.shape)
    print("Angle:", angle)
    print("Output Shape:", pred.shape)

    