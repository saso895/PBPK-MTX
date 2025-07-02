import numpy as np
import pandas as pd
from init_param import init_pars
from sklearn.model_selection import train_test_split
import os
# 生成或导入数据
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, 'ini_data', 'mei-MTX-09-10-once-final.csv')
df = pd.read_csv(csv_file_path)
#df = pd.read_csv('ini_data/mei-MTX-09-10-once-final.csv')
df = df.loc[:, ['ID', 'People','Time', 'Amount', 'DosingTime', 'Concentration']]
data = df.to_numpy()

# 提取每个个体的数据
time_points = []
concentration_data = []
input_dose = []
inject_timelen = []
input_dose_rate = []
ids = np.unique(data[:, 0])

for id in ids: 
    patient_data = data[data[:, 0] == id, :]
    
    arm_start_index = np.argwhere(patient_data[:, 2] == 0).flatten().tolist()

    arm_start_index.append(len(patient_data))
    for sindex, eindex in zip(arm_start_index[:-1], arm_start_index[1:]):
        if sindex+1 ==  eindex:
            aaa=1
        patient_in_dose = patient_data[sindex, 3]
        patient_in_timelen = patient_data[sindex, 4]
        time_points.append(patient_data[sindex:eindex, 2])
        concentration_data.append(patient_data[sindex:eindex, 5])
        input_dose.append(patient_in_dose)
        inject_timelen.append(patient_in_timelen)
        input_dose_rate.append(patient_in_dose / patient_in_timelen)
 
# 分割数据集为训练集和测试集，7:3比例
time_points_train, time_points_test, concentration_data_train, concentration_data_test, input_dose_train, input_dose_test, inject_timelen_train, inject_timelen_test, input_dose_rate_train, input_dose_rate_test = train_test_split(
    time_points, concentration_data, input_dose, inject_timelen, input_dose_rate, test_size=0.3, random_state=42
)

# 打印分割后的数据长度
print(f"训练集长度: {len(time_points_train)}")
print(f"测试集长度: {len(time_points_test)}")


