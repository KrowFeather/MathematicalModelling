from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
   #Load dataset
wine = datasets.load_wine()   
   # 显示数据集概况
print(wine.DESCR)

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) 
knn = KNeighborsClassifier(n_neighbors=5)  # K=5
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
   
   