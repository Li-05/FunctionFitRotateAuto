import numpy as np
import math
def getGrid(x_min, x_max, x_num, y_min, y_max, y_num):
    x = np.linspace(x_min, x_max, x_num)
    y = np.linspace(y_min, y_max, y_num)
    xx, yy = np.meshgrid(x, y)
    return xx.reshape((-1)), yy.reshape((-1))

# 旋转x, y点至0~beta度
def rotate_points(x, y, beta):
    angles = np.arctan2(y, x)  # 计算原始点的角度
    radii = np.sqrt(x**2 + y**2)  # 计算原始点的长度
    while np.any( angles <= 2 * np.pi ):
        angles = np.where(angles <= 2 * np.pi, angles+beta, angles) 
    a = radii * np.cos(angles)  # 计算旋转后的点的x坐标
    b = radii * np.sin(angles)  # 计算旋转后的点的y坐标
    return a, b

# 旋转不变性函数
def fitRotateFunc(x, y, beta = np.pi/3):
    X, Y = rotate_points(x, y, beta=beta)
    return np.sin(X+Y)

if __name__ == '__main__':
    x, y = getGrid(-1, 1, 2, -1, 1, 3)
    print(x)
    print(y)


