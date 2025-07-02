import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
OBSdata = pd.read_csv("./ini_data/mei0414point4.csv")

# 选择需要的列：Arm, Time, Amount, Rate, DosingTime, Concentration
Human_obs = OBSdata[['Arm', 'Time', 'Amount', 'Rate', 'DosingTime', 'Concentration']]

# 将 DataFrame 转换为 NumPy 数组
#data = Human_obs.to_numpy()
data = Human_obs

# 获取所有唯一的 Arm 组
arms = data['Arm'].unique()

# 初始化存储列表
rate_data = []
dosing_time_data = []
time_points = []
concentration_data = []
#arm_data = []
arm_list = []
# 遍历每个 Arm 组
for arm in arms:
    # 筛选出该 Arm 组的所有数据
    arm_data = data[data['Arm'] == arm]
    # 提取并存储每个 Arm 组的输注速率、输注时长、测量时间和浓度

    rate = arm_data.iloc[0, 3]  # 假设同一 Arm 组下 Rate 是固定的
    dosing_time = arm_data.iloc[0, 4]  # 假设同一 Arm 组下 DosingTime 是固定的

    # 提取时间点和浓度
    times = arm_data['Time'].values  # 使用列名选择列
    concentrations = arm_data['Concentration'].values  # 使用列名选择列
    # 将提取到的信息加入相应列表，确保每个 arm 组都有自己的记录
    rate_data.append(rate)
    dosing_time_data.append(dosing_time)
    time_points.append(times)
    concentration_data.append(concentrations)
    arm_list.append(arm)

# 使用 train_test_split 进行数据集的划分，设置 test_size 为 0.3 以便 7:3 划分
time_points_train, time_points_test, concentration_data_train, concentration_data_test, rate_data_train, rate_data_test, dosing_time_train, dosing_time_test, arm_data_train, arm_data_test = train_test_split(
    time_points, concentration_data, rate_data, dosing_time_data, arm_list, test_size=0.3, random_state=42
)

# 输出训练集和测试集长度，验证分割是否正确
print(f"训练集 Arm 组数量: {len(time_points_train)}")
print(f"测试集 Arm 组数量: {len(time_points_test)}")