import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

#Outcome = 1 Diyebet/Şeker Hastası
#Outcome = 0 Sağlıklı

data = pd.read_csv("diabetes.csv")
data.head()

diabetes = data[data.Outcome == 1]
healty = data[data.Outcome == 0]

plt.scatter(healty.Age, healty.Glucose, color='green', label='healty', alpha = 0.4)
plt.scatter(diabetes.Age, diabetes.Glucose, color='red', label='diabetes', alpha = 0.4)
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.legend()
plt.show()


# x ve y eksenlerini belirleyelim
y = data.Outcome.values
x_data = data.drop(['Outcome'], axis=1)
#Outcome sütünunu(depent variable) çıkarıp sadece independent variables bırakıyoruz
# Çünkü KNN algoritması x degerleri içerisinde gruplandırma yapacak..



# Normalization yapıyoruz - x_data içerisindeki degerler sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz
# Eğer bu şekilde normalization yapmazsak yüksek rakam küşük rakamları ezer ve KNN algoritması yanılta bilir!

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

print(x.head(), x_data.head())

# traij datamız ile test datamızı ayırıyoruz.
# train datamız sistemin sağlıklı insan ile hasta insan ayırt etmesini çğrenmek için kullanılacak
# test datamız ise bakalım makine öğrenme modelimiz doğru bir şekilde hasta ve sağlıklı insanları ayırt edebiliyor mu diye
# test etmek için kullanılacak...

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=1)

#KNN Modelimizi oluşturuyoruz.

knn = KNeighborsClassifier(n_neighbors=4) # n_neighbors = k
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print('K=4 için test verilerimizin doğrulama testi sonucu', knn.score(x_test, y_test))

# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
sc = MinMaxScaler()
sc.fit_transform(x_data)

new_prediction = knn.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))
new_prediction[0]

# k kaç olmalı ?
# en iyi k degerini belirleyelim..

count = 1
for k in range(1, 11):
    knn_new = KNeighborsClassifier(n_neighbors=k)
    knn_new.fit(x_train, y_train)
    print(count, '', 'Doğruluk oranı: %', knn_new.score(x_test, y_test)*100)
    count += 1

