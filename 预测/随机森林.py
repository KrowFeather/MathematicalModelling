import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def forest():
    # 加载数据集，读取本地的 Titanic 数据文件
    titan = pd.read_csv("titanic.csv")

    # 构造特征值和目标值
    # feature 包含乘客的 Pclass（舱位等级）、Age（年龄）、Fare（票价）、Sex（性别）
    # target 是 Survived（是否幸存），这是我们要预测的目标值
    feature = titan[["Pclass", "Age", "Fare", "Sex"]]
    target = titan["Survived"]

    # 特征预处理
    # 检查特征数据是否存在缺失值
    print(pd.isnull(feature).any())

    # 对 Age 列的缺失值进行填充
    # 使用 pop 方法将 Age 列从 feature 中取出并删除
    Age = feature.pop("Age")
    # 用 Age 列的均值填充缺失值
    Age = Age.fillna(Age.mean())
    # 将填充后的 Age 列重新插入到特征数据的第0列
    feature.insert(0, "Age", Age)

    # 字典特征抽取
    # 将特征数据转换为字典格式并通过 DictVectorizer 进行向量化处理
    dv = DictVectorizer()
    # 将特征字典转换为稀疏矩阵，并转换为数组格式
    feature = dv.fit_transform(feature.to_dict(orient="records"))
    feature = feature.toarray()
    # 打印转换后的特征数组和特征名称
    print(feature)
    print(dv.get_feature_names_out())

    # 划分数据集
    # 将数据集划分为训练集和测试集，测试集占 25%
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.25)
    # 打印训练集和测试集的维度信息
    print("训练集：", x_train.shape, y_train.shape)
    print("测试集：", x_test.shape, y_test.shape)

    # 创建随机森林分类器
    rf = RandomForestClassifier()

    # 超参数搜索
    # 定义超参数范围，n_estimators 表示树的数量，max_depth 表示树的最大深度
    param = {"n_estimators": [10, 20, 30, 40], "max_depth": [25, 35, 45]}
    # 使用 GridSearchCV 进行网格搜索，cv=5 表示5折交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=5)

    # 训练模型
    gc.fit(x_train, y_train)

    # 评估模型性能并打印结果
    # 打印模型在测试集上的准确率
    print("在测试集上的准确率：", gc.score(x_test, y_test))
    # 打印交叉验证中的最佳得分
    print("在验证集上的准确率：", gc.best_score_)
    # 打印最佳的模型参数组合
    print("最好的模型参数：", gc.best_params_)
    # 打印最优模型
    print("最好的模型：", gc.best_estimator_)


# 主程序入口，调用 forest 函数
if __name__ == "__main__":
    forest()
