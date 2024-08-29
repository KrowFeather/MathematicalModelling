import pandas as pd
from scipy.stats import mode
import numpy as np
def deal_name(file_path):
# 读取Excel文件  # 替换为你实际的文件路径
    df = pd.read_excel(file_path,engine='openpyxl')
    
    #数据格式化
    #df['评分'] = df['评分'].replace('--', np.nan)
    #df['评分'] = pd.to_numeric(df['评分'], errors='coerce')
    
    # 空缺数据众数化
    numeric_columns = df.select_dtypes(include=['number']).columns
    def fill_missing_with_mode(column):
        # 计算众数（mode）并将空缺值替换为众数
        print(mode(column.dropna()),mode(column.dropna())[0])
        try:
            most_frequent = mode(column.dropna())[0]
            return column.fillna(most_frequent)
        except:
            print("err happened")
            return -1
    df[numeric_columns] = df[numeric_columns].apply(fill_missing_with_mode)
    output_path=file_path
    df.sort_values('总分',inplace=True)
    df.to_excel(output_path,index=False)
# 定义一个函数来填充列中的空缺值为众
import os
for filename in os.listdir("./Data"):
    if(filename=='test01.py'):
        continue
    print(filename)
    deal_name(filename)

#deal_name("佛山.xlsx")