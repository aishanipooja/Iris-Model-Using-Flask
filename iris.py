import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

df = pd.read_csv('iris.data')
x = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
y = labelencoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
svc = SVC(kernel='linear').fit(x_train,y_train)

pickle.dump(svc, open('iri.pkl', 'wb'))

