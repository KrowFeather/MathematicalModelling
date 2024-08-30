import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
def prework():
    titan=pd.read_csv('titanic.csv')
    #print(titan)
    feature=titan[["Pclass","Age","Fare","Sex"]]
    #print(feature)
    target=titan["Survived"]
    #print(target)
    print(pd.isnull(feature).any())
    
    Age=feature.pop("Age")
    Age=Age.fillna(Age.mean())
    feature.insert(0,"Age",Age)
    to_dict(feature)

def to_dict(feature):
   dv=DictVectorizer() 
   feature=dv.fit_transform(feature.to_dict(orient="records"))
   feature=feature.toarray()
   print(feature)
   print(dv.get_feature_names_out())

prework()

