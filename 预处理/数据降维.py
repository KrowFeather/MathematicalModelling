import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 生成一些示例数据
data = {
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(5, 2, 100),
    'C': np.random.normal(-5, 3, 100),
    'D': np.random.normal(10, 4, 100),
    'E': np.random.normal(-10, 5, 100)
}
df = pd.DataFrame(data)
print(df)
# 标准化数据（PCA需要标准化数据）
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 使用PCA进行降维
pca = PCA(n_components="mle")  # 设置PCA的降维目标根据我们的最大似然估计自动设置
# pca=PCA(n_components=0.97,svd_solver="full")
df_pca = pca.fit_transform(df_scaled)  # 对标准化后的数据进行PCA降维

# 打印PCA降维后的数据
print("PCA降维后的数据：")
print(df_pca)
exit(0)
# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)  # 设置t-SNE的降维目标为2维，并设置随机种子
df_tsne = tsne.fit_transform(df_scaled)  # 对标准化后的数据进行t-SNE降维

# 打印t-SNE降维后的数据
print("t-SNE降维后的数据：")
#print(df_tsne)

# 绘制PCA和t-SNE降维后的数据
plt.figure(figsize=(12, 6))

# 绘制PCA降维后的数据
plt.subplot(1, 2, 1)
plt.scatter(df_pca[:, 0], df_pca[:, 1])
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 绘制t-SNE降维后的数据
plt.subplot(1, 2, 2)
plt.scatter(df_tsne[:, 0], df_tsne[:, 1])
plt.title('t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.show()