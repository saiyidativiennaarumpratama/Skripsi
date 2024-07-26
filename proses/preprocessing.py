import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


df = pd.read_excel("data/DatasetDiabetesMelitus_Skripsi.xlsx")

def OneHot():
    # One Hot Encoding Jenis Kelamin
    st.subheader('Data Sebelum Proses One Hot Encoding')
    st.write(df)
    encoder = OneHotEncoder()
    hasil_encoder = encoder.fit_transform(df[["JK"]]).toarray()

    # Menggabungkan hasil one-hot encoding dengan DataFrame asli dan menghapus kolom "JK"
    dfonehot = pd.concat([df.drop(["JK"], axis=1), pd.DataFrame(hasil_encoder, columns=['JK_L', 'JK_P'])], axis=1)
    st.subheader('Data Setelah Proses One Hot Encoding')
    st.session_state['dfonehot'] = dfonehot
    st.write(dfonehot)

def LabelEncoder():
    if 'dfonehot' in st.session_state:
        dflabelencoder = st.session_state['dfonehot']
        label_encoder = preprocessing.LabelEncoder()
        dflabelencoder['Diagnosa']= label_encoder.fit_transform(dflabelencoder['Diagnosa'])
        st.session_state['dflabelencoder'] = dflabelencoder
        st.write(dflabelencoder)
        # joblib.dump(dflabelencoder, 'preprocessed_data.pkl')
        # dflabelencoder['Diagnosa'].unique()
        # dfbaru['Diagnosa'].value_counts()
    
def MinMax():
    st.subheader('Data Sebelum Proses Normalisasi')
    if 'dflabelencoder' in st.session_state:
        dfminmax = st.session_state['dflabelencoder']
        st.write(dfminmax)
        scaler = MinMaxScaler()
        columns_to_normalize = ['Usia',	'Sistolik',	'Diastolik',	'Napas',	'Nadi',	'GDA',	'Cholesterol_LDL',	'Triglyserida']

        # Melakukan Min-Max Scaling pada kolom-kolom tersebut
        dfminmax[columns_to_normalize] = scaler.fit_transform(dfminmax[columns_to_normalize])
        st.subheader('Data Setelah Proses Normalisasi')
        st.session_state['dfminmax'] = dfminmax
        st.write(dfminmax)

def Imputasi():
    st.subheader('Data Sebelum Proses Imputasi Missing Value')
    if 'dfminmax' in st.session_state:
        dfimputasi = st.session_state['dfminmax']
        st.write(dfimputasi)

        # cek nilai mean pada fitur kolesterol
        mean_kolesterol = dfimputasi['Cholesterol_LDL'].mean()
       
        # cek nilai mean pada fitur triglyserida
        mean_triglyserida = dfimputasi['Triglyserida'].mean()
        
        # Mengisi missing value dengan mean pada masing-masing fitur
        dfimputasi['Cholesterol_LDL'].fillna(mean_kolesterol, inplace=True)
        dfimputasi['Triglyserida'].fillna(mean_triglyserida, inplace=True)

        # Menampilkan DataFrame setelah pengisian missing value
        st.subheader('Data Setelah Imputasi Missing Value')
        st.session_state['dfimputasi'] = dfimputasi
        st.write(dfimputasi)

# Pembagian Dataset
def Oversampling():
    if 'dfimputasi' in st.session_state:
        split_data = st.session_state['dfminmax']
        X = np.array(split_data[['Usia',	'Sistolik',	'Diastolik',	'Napas',	'Nadi',	'GDA',	'Cholesterol_LDL',	'Triglyserida', 'JK_L',	'JK_P']])
        y = np.array(split_data['Diagnosa'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # Melakukan oversampling hanya pada data train
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        # Menampilkan jumlah data sebelum dan sesudah oversampling dalam satu diagram batang
        fig, ax = plt.subplots()
        counts_before = pd.Series(y_train).value_counts().sort_index()
        counts_after = pd.Series(y_train_resampled).value_counts().sort_index()

        bar_width = 0.15
        index = range(len(counts_before))

        bar1 = ax.bar(index, counts_before, bar_width, color='lightblue', label='Sebelum SMOTE')
        bar2 = ax.bar([i + bar_width for i in index], counts_after, bar_width, color='pink', label='Sesudah SMOTE')

        ax.set_xlabel('Kelas')
        ax.set_ylabel('Jumlah')
        ax.set_title('Jumlah Data Sebelum dan Sesudah Oversampling')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(['Tidak Komplikasi', 'Komplikasi'])
        ax.legend()

        # Menambahkan angka di atas batang
        for bar in bar1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, '%d' % int(height), ha='center', va='bottom', color='black')

        for bar in bar2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, '%d' % int(height), ha='center', va='bottom', color='black')
        st.pyplot(fig)

    




   

    


