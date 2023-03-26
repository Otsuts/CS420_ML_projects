import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义 x 轴、y 轴和 z 轴数据
x = np.arange(5)
y = np.arange(3)
z2 = np.array([[-2.8561, -6.1951, -9.6988, -2.8358,-6.7997], [-10.4656, -2.8377, -6.9208, -10.8257, -2.8503], [-6.9274, -10.8829, -2.8638, -7.0067, -11.0924]])+12.0
z3 = np.array([[-2.6896,-6.1951 , -9.6988, -2.7675, -6.6766], [-10.4656, -2.7870, -6.8338,-10.8131,  -2.7966], [-6.8721, -10.8829, -2.7539,  -6.9572, -11.0056]])+12.0
z1 = np.array([[-2.8561, -6.9964, -10.1792, -2.8358, -7.1131], [-11.0954, -2.8857, -7.0545, -11.0714, -2.8955], [-7.0206, -11.4522, -2.8638, -7.1290, -11.1941]])+12.0

# 创建画布和 3D 坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置 x 轴和 y 轴的网格
xx, yy = np.meshgrid(x, y)

# 将 x, y, z1, z2, z3 展平为 1 维数组
xx = xx.ravel()
yy = yy.ravel()
z1 = z1.ravel()
z2 = z2.ravel()
z3 = z3.ravel()

# 设置柱状图的参数
dx = dy = 0.2  # 柱体的宽度和深度
dz1 = z1  # 第一组数据的柱体高度
dz2 = z2  # 第二组数据的柱体高度
dz3 = z3  # 第三组数据的柱体高度

# 绘制柱状图
ax.bar3d(xx, yy, np.zeros(len(z1)), dx, dy, dz1, color='b')
ax.bar3d(xx+0.2, yy, np.zeros(len(z1)), dx, dy, dz2, color='g')
ax.bar3d(xx+0.4, yy, np.zeros(len(z1)), dx, dy, dz3, color='r')

# 设置坐标轴标签和范围
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 5)
ax.set_ylim(0, 3)
ax.set_zlim(0, 30)

# 显示图形
plt.show()
