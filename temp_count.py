import scipy.io
import numpy as np

# 加载mat文件
mat_data = scipy.io.loadmat('Dataset/subject1/test.mat')

# 提取Y矩阵
Y = mat_data['Y'].flatten()  # 将Y矩阵展平成1维数组

# 统计每个标签的数量
count_2 = np.sum(Y == 2)
count_3 = np.sum(Y == 3)
count_7 = np.sum(Y == 7)

# 输出结果
print(f"标签2的数量: {count_2}")
print(f"标签3的数量: {count_3}")
print(f"标签7的数量: {count_7}")
