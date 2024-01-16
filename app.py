import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def loadData():
    file_path = 'processedCleveland.data'
    dataset = pd.read_csv(file_path)
    return dataset

# Basic preprocessing required for all the models.
def preprocessing(dataset):
    dataset.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']
    dataset.replace({"?": np.nan}, inplace=True)
    dataset = dataset.astype('float64', errors='ignore')  # Ubah ke float64 untuk menangani nilai NaN
    dataset['target'] = dataset.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    dataset['thal'].fillna(dataset['thal'].median(), inplace=True)
    dataset['ca'].fillna(dataset['ca'].median(), inplace=True)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

# Training K-NN Classifier
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def kNN(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Inisialisasi K-NN Classifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

    # Fitting K-NN Classifier to the Training set
    classifier.fit(X_train, y_train)

    # Prediksi menggunakan K-NN Classifier
    y_pred = classifier.predict(X_test)

    score1 = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score1, report, classifier

def prediction(name, age, sex, cp, tresthbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, cs, thal):
    data = loadData()
    X_train, X_test, y_train, y_test = preprocessing(data)
    score1, report, classifier = kNN(X_train, X_test, y_train, y_test)
    prediksi = classifier.predict(sc.transform([[age, sex, cp, tresthbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, cs, thal]]))
    prediksi_perc = int(prediksi[0] * 100)
    st.write("Saudara ", name, " Memiliki kemungkinan memiliki penyakit jantung sebesar ", prediksi_perc, "%")

def main():
    st.title("PREDIKSI PENYAKIT JANTUNG (K-NN)")
    st.markdown("Aplikasi ini dibuat untuk memprediksi kemungkinan seseorang mengidap penyakit jantung menggunakan algoritma K-NN")
    data = loadData()
    X_train, X_test, y_train, y_test = preprocessing(data)
    score1, report, classifier = kNN(X_train, X_test, y_train, y_test)
    name = st.text_input("Masukkan nama anda :", 'nama')
    age = st.number_input("Masukkan usia anda", 0, 100, 0)
    pilih_sex = st.radio("Pilih jenis kelamin ", options=['Laki-laki', 'Perempuan'])
    sex = 1 if pilih_sex == 'Laki-laki' else 0
    display_cp = st.checkbox("Apakah anda mengalami sakit di dada ? *centang jika iya")
    cp = st.radio("Pilih jenis sakit yang anda rasakan : ", options=['Typical Angina', 'Atypical angina', 'Non-anginal pain', 'Asymptotic'], index=0)
    tresthbp = st.number_input("Masukkan tekanan darah anda (/mmHg) : ", 50, 400, 120)
    chol = st.number_input("Masukkan kadar kolesterol darah anda (mg/dL) :", 100, 400, 180)
    pilih_fbs = st.checkbox("Apakah anda memiliki gula daran lebih dari 120mg/dL ? Centang jika iya")
    fbs = 1 if pilih_fbs else 0
    pilih_restecg = st.radio("Pilih hasil elektrokardiografi anda : ", options=['Normal', 'ST-T wave abnormality', 'Left ventricular hyperthroph'], index=0)
    restecg = 0 if pilih_restecg == 'Normal' else 1 if pilih_restecg == 'ST-T wave abnormality' else 2
    thalach = st.number_input("Masukkan tekanan darah tertinggi anda (/mmHg) : ", 50, 400, 120)
    pilih_exang = st.checkbox("Apakah ketika melakukan aktivitas fisik berat seperti olahraga, dada anda terasa sakit ? Centang jika iya")
    exang = 1 if pilih_exang else 0
    oldpeak = st.number_input("Masukkan nilai ST Depression : ", 0, 10, 1)
    pilih_slope = st.radio("Masukkan bentuk kurva ST ", options=['Unsloping', 'Flat', 'Downsloping'], index=0)
    slope = 1 if pilih_slope == 'Unsloping' else 2 if pilih_slope == 'Flat' else 3
    cs = st.number_input("Masukkan pembuluh darah utama diwarnai dengan fluoroskopi", 0, 3, 1)
    pilih_thal = st.radio("Kelainan Thalassemia ", options=['Normal', 'Fixed defect', 'reversible defect'], index=0)
    thal = 3 if pilih_thal == 'Normal' else 6 if pilih_thal == 'Fixed defect' else 7

    st.write(name, age, sex, cp, tresthbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, cs, thal)
    butt = st.button('Prediksi')
    if butt:
        prediction(name, age, sex, cp, tresthbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, cs, thal)

if __name__ == "__main__":
    main()