import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# veri seti okuma islemi
dataset = pd.read_csv('creditcard.csv')
print(dataset.to_string())

# veri setindeki kolonlarin isimlerini degistirme islemi
print(dataset.columns)
new_columns =["musteri_seri_no","kredikart_parola","limit","kredikart_sayisi,",
              "toplam_ziyaret","toplam_cevrimici_ziyaret","toplam_arama"]

for i,c in zip(dataset.columns, new_columns):

  dataset.rename(columns = {i:c}, inplace = True)

print(dataset.columns)

print(dataset.to_string())

# veri setine dair kesifsel analiz

# veri seti boyutunu ogrenmek
print("\n\nVeri seti boyut : ",dataset.shape)

print("\n\n")

# veri setine dair bilgi almak
print(dataset.info())

print("\n\n")

# veri setine dair bilgi almak
print(dataset.describe().to_string())

print("\n\n")

# veri setinde tekrarlayan verileri kontrol etmek
print("Veri setinden satirlarda tekrar eden veri olan satir sayisi : ",dataset.duplicated().sum())

print("\n\n")

# veri setinde eksik olan verileri kontrol etmek
print("Veri setinde satirlarda eksik veri olan kayit sayisi :\n")
print(dataset.isnull().sum())

# cikarilan bilgiler

# veri setinin 7 kolonu var 685 satiri var
# kolonlarda, verilerin minimum ve maksimum degerleri normal burada bir sorun yok
# veri setinde veri tiplerinde bir sorun yok bunlar da normal

# YAPILACAK VERI ON ISLEME CALISMALARI

#1) kolonlardan 2 tanesi modelimizde gereksiz yer teskil ediyor. (SILINECEK)
#2) 16 adet tekrarlayan veri var. (SILINECEK)
#3) 16 adet eksik veri var. (SILINECEK)
#4) Veri setindeki verileri birbirinden orantısız duruyor. (NORMALIZASYON YAPILACAK)


# tekrarlayan veriler siliniyor.

print("Veri setinden satirlarda tekrar eden veri olan satir sayisi : ",dataset.duplicated().sum())
new_dataset = dataset.drop_duplicates()

# silinme sonrasi kontrol
print("Veri setinden satirlarda tekrar eden veri olan satir sayisi : ",new_dataset.duplicated().sum())


# eksik olan veriler siliniyor.

print("Veri setinde satirlarda eksik veri olan kayit sayisi :\n")
print(new_dataset.isnull().sum())

real_dataset = new_dataset.dropna()

print("\n\n")
print("Veri setinde satirlarda eksik veri olan kayit sayisi :\n")
print(real_dataset.isnull().sum())


# modele uymayan gereksiz veriler veri setinden siliniyor.

real_dataset.drop(columns = ['musteri_seri_no', 'kredikart_parola'], inplace = True)
print(real_dataset.describe().to_string())


# korelasyon ısı haritası gösteriliyor
corrmat= real_dataset.corr()
plt.figure(figsize=(10,7))
plt.title('Correlation Map')
sns.heatmap(corrmat,annot=True)
plt.show()

# korelasyon matrisinde birbirine çok yakın ilişki kolonlar yok.


################################################################################
# K-MEANS ALGORİTMASI KULLANIMI

# Öncelikle k degerini buluyoruz.

from sklearn.cluster import KMeans

# elbow yontemi ile k degerini bulmaya calisiyoruz.
sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(real_dataset)
    sonuclar.append(kmeans.inertia_)  #inertia wcss değerleri

plt.plot(range(1,11),sonuclar)
plt.title('Elbow Method')
plt.show()


# fotografta k=3 icin bir dirsek noktasi oldugunu goruyoruz. bu nedenle k degerimizi 3 olarak aliyoruz.
# k = 3


# GORSELLESTIRMEK ICIN VERI SETINI 2 KOLONA DUSURUYORUZ.

x1 = real_dataset.iloc[:, 1]
x2 = real_dataset.iloc[:, 3]

data = pd.concat([x1, x2], axis=1)

X = data.iloc[:, 0:2].values


##########

# ELBOW YONTEMI ILE BULDUGUMUZ K DEGERI ICIN ALGORITMAYI KULLANIYORUZ.

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Küme 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Küme 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Küme 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'Küme Merkezleri')
plt.title('K-MEANS GRAFİK')
plt.xlabel('Kart Sayısı')
plt.ylabel('Çevrimiçi Ziyaret')
plt.legend()
plt.show()

################################################################################

# HC ALGORİTMASI

from sklearn.cluster import AgglomerativeClustering


# ELBOW YONTEMI ILE BULDUGUMUZ K = 3 DEGERINI BURADA DA KULLANIYORUZ

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

y_predict = ac.fit_predict(X)

plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100, c='red', label = 'Küme 1')
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100, c='blue', label = 'Küme 2')
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100, c='green', label = 'Küme 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'Küme Merkezleri')
plt.title('HC')
plt.xlabel('Kart Sayısı')
plt.ylabel('Çevrimiçi Ziyaret')
plt.legend()

plt.show()



# SOM ALGORİTMASI KULLANIMI

from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM

som = SOM(m=3, n=1, dim=2, random_state=123)
som.fit(X)
predictions = som.predict(X)


x = X[:,0]
y = X[:,1]

colors = ['red', 'blue', 'green']
plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'orange', label = 'Küme Merkezleri')
plt.title('SOM GRAFİK')
plt.xlabel('Kart Sayısı')
plt.ylabel('Çevrimiçi Ziyaret')
plt.legend()

plt.show()
