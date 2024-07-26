import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras
from scipy.stats import mode
# tf.compat.v1.reset_default_graph()
def backpro():
    if 'dfimputasi' in st.session_state:
        split_data = st.session_state['dfminmax']
        X = np.array(split_data[['Usia',	'Sistolik',	'Diastolik',	'Napas',	'Nadi',	'GDA',	'Cholesterol_LDL',	'Triglyserida', 'JK_L',	'JK_P']])
        y = np.array(split_data['Diagnosa'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        #### model ####
        loaded_model = keras.models.load_model('model/Skenario1_Epoch_model_neuron_5_learning_rate_0.1_epochs_300.h5')
        # Prediksi menggunakan model yang sudah di-load
        y_pred = loaded_model.predict(X_test)

        # Konversi probabilitas menjadi kelas biner dengan threshold 0.5
        threshold = 0.5
        y_pred_class = (y_pred >= threshold).astype(int)

        # Hitung akurasi pada data testing
        akurasi = accuracy_score(y_test, y_pred_class)
        st.write("Hasil Akurasi:", akurasi)
        st.subheader('Confussion Matrix')

        cm = confusion_matrix(y_test, y_pred_class)

        # Buat plot menggunakan matplotlib
        fig_cm = plt.figure(figsize=(3, 2))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 10})

        # Tambahkan label dan judul
        plt.xlabel('Predicted labels', fontsize=7)
        plt.ylabel('True labels', fontsize=7)
        plt.title('Confusion Matrix', fontsize=7)
        st.pyplot(fig_cm)

def backpro_bagging():
    if 'dfimputasi' in st.session_state:
        split_data = st.session_state['dfminmax']
        Xb = np.array(split_data[['Usia',	'Sistolik',	'Diastolik',	'Napas',	'Nadi',	'GDA',	'Cholesterol_LDL',	'Triglyserida', 'JK_L',	'JK_P']])
        yb = np.array(split_data['Diagnosa'])
        X_train, Xb_test, y_train, yb_test = train_test_split(
            Xb, yb, test_size=0.2, random_state=42)
        
        ###### LOAD MODEL ######
        # Fungsi untuk memuat semua model yang disimpan
        # def load_all_models(n_estimator):
        #     models = []
        #     for i in range(n_estimator):
        #         # D:\Kuliah\Skripsi - Fix\Streamlit\model
        #         model_path = f'/model/skenario2_model_{n_estimator}_{i+1}.h5'
        #         model = load_model(model_path)
        #         models.append(model)
        #     return models
        # Fungsi untuk memuat semua model yang disimpan di folder 'model'
        def load_all_models(n_estimator):
            import os
            models = []
            model_folder = 'model'  # Folder tempat model disimpan
            for i in range(n_estimator):
                model_path = os.path.join(model_folder, f'skenario2_model_{n_estimator}_{i+1}.h5')
                try:
                    model = keras.models.load_model(model_path)
                    models.append(model)
                except Exception as e:
                    st.error(f"Error loading model {model_path}: {e}")
            return models
        # Fungsi untuk melakukan prediksi dengan semua model dan melakukan majority voting
        def majority_voting_prediction(models, Xb_test):
            all_predictions = []
            for model in models:
                y_pred = model.predict(Xb_test)
                y_pred_class = (y_pred >= 0.5).astype(int)
                all_predictions.append(y_pred_class)
            
            # Melakukan majority voting
            majority_vote = mode(np.array(all_predictions), axis=0)[0].flatten()
            return majority_vote

        # Load semua model yang telah disimpan
        n_models = 9
        models = load_all_models(n_models)

        # Melakukan prediksi dengan data testing
        majority_vote_test = majority_voting_prediction(models, Xb_test)

        # Hitung akurasi pada data testing
        akurasi_test = accuracy_score(yb_test, majority_vote_test)
        st.write("Hasil Akurasi :", akurasi_test)
        st.subheader('Confussion Matrix')

        cm = confusion_matrix(yb_test, majority_vote_test)

        # Buat plot menggunakan matplotlib
        fig_cm = plt.figure(figsize=(3, 2))
        sns.heatmap(cm, annot=True, cmap='RdPu', fmt='g', annot_kws={"size": 10})

        # Tambahkan label dan judul
        plt.xlabel('Predicted labels', fontsize=7)
        plt.ylabel('True labels', fontsize=7)
        plt.title('Confusion Matrix', fontsize=7)
        st.pyplot(fig_cm)

        # Pastikan akurasi ini sama dengan akurasi sebelumnya

        
