import torch
import matplotlib.pyplot as plt
from net import *
from util import *
from torch.utils.data import DataLoader
from data import *
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = "./param/func.pth"
beta = np.pi/3
net = CombinedNet().to(device)
net.load_state_dict(torch.load(weight_path))
net.eval()  # 将模型设为评估模式

data_loader = DataLoader(MyDataset(width=80, height=80, beta=beta), batch_size=1, shuffle=False)

x_array = []
y_array = []
_z_array = []
z_array = []
content_tensor = getContentTensor().unsqueeze(0).to(torch.float32).to(device)
start_time = time.time()
for i, (xy, z) in enumerate(data_loader):
    xy = xy.to(device)
    out = net(xy)
    _z = out[0].cpu().item()
    angle = out[1].cpu().item()

    x = xy[0][0].cpu().item()
    y = xy[0][1].cpu().item()
    x_array.append(x)
    y_array.append(y)
    _z_array.append(_z)
    z_array.append(z.item())
end_time = time.time()

mse = np.mean((np.array(_z_array) - np.array(z_array)) ** 2)
print("MSE:", mse)

inference_time = end_time - start_time
print("Inference time:", inference_time, "seconds")

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x_array, y_array, _z_array, c=_z_array, cmap='viridis')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.view_init(elev=30, azim=120)  # 设置视角
ax1.set_title('Model Fitten')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_array, y_array, z_array, c=z_array, cmap='viridis')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
ax2.view_init(elev=30, azim=120)  # 设置视角
ax2.set_title('Ground Truth')
plt.savefig('./param/fit_curve.png', dpi=300, bbox_inches='tight')
