import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 生成一些示例数据
data = {
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(5, 2, 100),
    'C': np.random.normal(-5, 3, 100)
}
df = pd.DataFrame(data)
print(df)
# 定义一个函数来识别和处理异常值
def identify_and_handle_outliers(df, column):
    # 计算四分位数和四分位距
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # 定义异常值的阈值
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 识别异常值
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # 处理异常值（例如，用中位数替换）
    median_value = df[column].median()
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = median_value
    
    return outliers

# 处理每个列的异常值
outliers_dict = {}
for column in df.columns:
    outliers = identify_and_handle_outliers(df, column)
    outliers_dict[column] = outliers

# 打印异常值
for column, outliers in outliers_dict.items():
    print(f"Outliers in column {column}:")
    print(outliers)
    print()

# 绘制箱线图
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title('Boxplot of Data with Outliers Handled')
plt.show()