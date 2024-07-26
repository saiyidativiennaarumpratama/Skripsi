import pickle
import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from proses.preprocessing import OneHot
from proses.preprocessing import MinMax
from proses.preprocessing import Imputasi
from proses.preprocessing import LabelEncoder
from proses.preprocessing import Oversampling
from proses.model import backpro
from proses.model import backpro_bagging
from proses import implementasi
from proses.implementasi import ujibackpro
# from proses import backpro
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
st.title('Klasifikasi Penyakit Diabetes Melitus')
# Garis horizontal
st.markdown("""
    <style>
        .rainbow-divider {
            height: 4px;
            background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
            border: none;
        }
    </style>
    <hr class="rainbow-divider">
    """, unsafe_allow_html=True)

# navigasi sidebar
with st.sidebar :
    selected = option_menu ('Menu',
                        ['Home', 'Data Preprocessing', 'Klasifikasi', 'Uji Coba'])


# Halaman Home
if (selected == 'Home') :
    st.subheader('Diabetes Melitus')
    st.divider()
    text = """Diabetes Melitus (DM) ialah penyakit yang melibatkan gangguan pada metabolisme tubuh yang ditandai dengan tingginya kadar glukosa dalam darah, yang dikenal sebagai hiperglikemia. Hiperglikemia adalah kadar glukosa yang tinggi dalam darah. Diagnosa pada penyakit DM ditegakkan atas dasar pemeriksaan kadar glukosa dalam darah. Kriteria untuk menentukan penyakit diabetes melitus dapat dilihat dari tes gula darah dan beberapa tes yang dapat menentukan tingkat gula dalam darah. Seperti pengecekan gula darah sewaktu jika hasil menunjukkan >200 mg/dL (11,1 mmol/L). Lalu Pengecekan gula darah puasa menunjukkan >126 mg/dL (> 7.0 mmol/L). Pada pasien Diabetes Melitus keadaan hiperglikemia kronis dapat menyebabkan berbagai komplikasi baik mikrovaskular maupun makrovaskular. Pasien diabetes tipe 2, baik yang sudah lama maupun baru terdiagnosis, berisiko mengalami komplikasi. Komplikasi makrovaskular umumnya menyerang jantung, otak, dan pembuluh darah. Sementara komplikasi mikrovaskular dapat terjadi pada mata dan ginjal."""
    st.markdown(f"""
    <div style="text-align: justify;">
        {text}
    </div>
    """, unsafe_allow_html=True)
    # isi penjelasan tentang diabet

    st.subheader('View Dataset')
    st.write('Data yang digunakan adalah data penyakit diabetes dari RSUD Sumberrejo Bojonegoro')
    df_X = pd.read_excel("data/DatasetDiabetesMelitus_Skripsi.xlsx")
    df_y = df_X['Diagnosa']
    if (selected == 'Home'):
        st.success(
            f"Jumlah Data : {df_X.shape[0]} Data, dan Jumlah Fitur : {df_X.shape[1]} Fitur")
        dataframe, keterangan = st.tabs(['Dataset', 'Keterangan'])
        with dataframe:
            st.write(df_X)
        with keterangan:
            st.text("""
                    - Jenis Kelamin : Menunjukkan jenis kelamin laki-laki dan Perempuan (L/P).
                    - Usia : Menunjukkan umur pasien dalam satuan tahun.
                    - Sistolik : Menunjukkan hasil pemeriksaan darah Sistolik dalam satuan mm/Hg. 
                    - Diastolik : Menunjukkan hasil pemeriksaan darah Diastolik dalam satuan mm/Hg.
                    - Napas : Menunjukkan hasil pemeriksaan frekuensi napas per menit.
                    - Nadi : Menunjukkan hasil pemeriksaan denyut nadi per menit.
                    - GDA: Menunjukkan hasil pemeriksaan gula darah acak pasien yang dilakukan kapan saja tanpa perlu berpuasa atau mempertimbangkan kapan terakhir waktu makan. 
                    - Cholesterol LDL : Menunjukkan hasil pemeriksaan kadar kolesterol LDL (Low Density Lipoprotein)  pasien dengan satuan mg/dL.
                    - Triglyserida : Menunjukkan hasil pemeriksaan kadar triglyserida pasien dengan satuan mg/dL.
                    """)


elif (selected == 'Data Preprocessing') :
    st.subheader('Data Preprocessing')
    st.divider()
    oneHot, labelencoder, normalisasi, imputasi, oversampling = st.tabs(['One Hot Encoding', 'Label Encoder', 'Normalisasi', 'Imputasi Missing Value', 'Penyeimbangan Data'])
    with oneHot:
        OneHot()

    with labelencoder:
        LabelEncoder()

    with normalisasi:
        MinMax()
    with imputasi:
        Imputasi()

    with oversampling:
        Oversampling()


elif (selected == 'Klasifikasi') :
    backpropagation, backpropagation_bagging = st.tabs(['Backpropagation', 'Backpropagation&Bagging'])
    with backpropagation :
        st.header('Hasil Klasifikasi Menggunakan Backpropagation')
        param, value = st.columns(2)
        with param :
            'Optimizer',':', 'Adam'
            'Fungsi Aktivasi',':', 'Sigmoid'
            'Learning Rate', ':', '0.1'
            'Epoch',':', '300'
        with value :
            'Neuron Input Layer', ':',  '10'
            'Neuron Hidden Layer',':', '5'
            'Neuron Output Layer' , ':', '1'
        backpro()
    with backpropagation_bagging:
        st.header('Hasil Klasifikasi Menggunakan Backpropagation&Bagging')
        param, value = st.columns(2)
        with param :
            'Optimizer',':', 'Adam'
            'Fungsi Aktivasi',':', 'Sigmoid'
            'Learning Rate', ':', '0.1'
            'Epoch',':', '300'
        with value :
            'Neuron Input Layer', ':',  '10'
            'Neuron Hidden Layer',':', '5'
            'Neuron Output Layer' , ':', '1'
            'n_estimator', ':', '9'
        backpro_bagging()
        
        
    

elif (selected == 'Uji Coba') :
    st.subheader('Uji Coba Klasifikasi')
    st.divider()
    col1, col2 = st.columns(2)
    with st.form(key='prediksi_form'):
        with col1:
            Usia = st.number_input('Input Usia')
            Sistolik = st.number_input('Input Tekanan Darah Sistolik')
            Diastolik = st.number_input('Input Tekanan Darah Diastolik')
            Napas = st.number_input('Input Frekuensi Napas')       

        # Mengisi kolom kedua
        with col2:
            Nadi = st.number_input('Input Detak Nadi')
            GDA = st.number_input('Input GDA')
            Cholesterol_LDL = st.number_input('Input Kadar Cholesterol LDL')
            Triglyserida = st.number_input('Input Kadar Triglyserida')
            # JK_P = st.number_input('jnis klmnin')
            # JK_L = st.number_input('jk')

        JK = st.selectbox('JK (Jenis Kelamin)', ('Laki-Laki', 'Perempuan'))
        if JK == 'Laki-Laki':
            JK_L = 1
            JK_P = 0
        else:
            JK_L = 0
            JK_P = 1
        # option = st.selectbox(
        # "Jenis Kelamin",
        # ("Laki-Laki", "Perempuan"))

        # prediksi = st.form_submit_button(label="Prediksi", type="primary")
        prediksi = st.form_submit_button(label="Prediksi")
        if prediksi :
            # data_input = [[Usia, Sistolik, Diastolik, Napas, Nadi, GDA, Cholesterol_LDL, Triglyserida, JK]]
            data_input = [[Usia, Sistolik, Diastolik, Napas, Nadi, GDA, Cholesterol_LDL, Triglyserida, JK_P, JK_L]]
            scaler = MinMaxScaler()
            # columns = ['Usia',	'Sistolik',	'Diastolik',	'Napas',	'Nadi',	'GDA',	'Cholesterol_LDL',	'Triglyserida']
            data_input_normalized = scaler.fit_transform(data_input)

            prediction = implementasi.ujibackpro(data_input_normalized)
            

            # st.button("Hasil Prediksi")
            # Lakukan one-hot encoding pada data input
            # data_input_encoded = OneHot(data_input)
            # # Lakukan label encoding pada data input
            # data_input_lblenco = LabelEncoder(data_input_encoded)
            # # Lakukan normalisasi pada data input
            # data_input_normalized = MinMax(data_input_lblenco)

            # prediction = implementasi.ujibackpro(data_input)
            # Tampilkan hasil prediksi
            # st.write(f'Hasil Prediksi: {prediction}')


    # st.button("Hasil Prediksi")




# if (menu == 'Home') :
#     st.write('Website Klasifikasi Penyakit Diabetes Melitus')
