import numpy as np
from scipy.io import loadmat, savemat

# 读取 true_labels.txt 文件中的数据
with open('true_label.txt', 'r') as file:
    data = file.readlines()

# 将数据转换为 numpy 数组并将其变成列向量
Y = np.array([float(x.strip()) for x in data]).reshape(-1, 1)

# 尝试加载现有的 .mat 文件，如果文件不存在则创建一个空字典
try:
    mat_data = loadmat('test.mat')
except FileNotFoundError:
    mat_data = {}

# 将新的 Y 矩阵添加到 mat 数据中
mat_data['Y'] = Y

# 将更新后的数据保存回 .mat 文件中
savemat('test.mat', mat_data)

print(f'Data has been successfully saved to test.mat as the Y matrix, with original data preserved.')
