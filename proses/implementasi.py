import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import tensorflow as tf
import keras
# tf.compat.v1.reset_default_graph()
# diagnosa = [0, 1]
# diagnosa = ['Komplikasi', 'Tidak Komplikasi']
def ujibackpro(data_baru):
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
    def majority_voting_prediction(models, data_baru):
            data_baru = np.array(data_baru)
            all_predictions = []
            for model in models:
                y_pred = model.predict(data_baru)
                y_pred_class = (y_pred >= 0.5).astype(int)
                all_predictions.append(y_pred_class)
            
            # Melakukan majority voting
            majority_vote = mode(np.array(all_predictions), axis=0)[0].flatten()
            return majority_vote
    # Menentukan jumlah model yang akan dimuat
    n_estimator = 9

    # Memuat semua model yang sudah disimpan
    models = load_all_models(n_estimator)

    # Melakukan prediksi dengan majority voting untuk data input baru
    y_new_pred = majority_voting_prediction(models, data_baru)
    if y_new_pred == 0:
        st.write('Komplikasi')
    elif y_new_pred == 1:
        st.write('Tidak Komplikasi')

    # Menampilkan hasil prediksi
    # print("Prediksi untuk data baru:")
    # print(y_new_pred)



    ########
    # model = keras.models.load_model('model/Skenario1_Epoch_model_neuron_5_learning_rate_0.1_epochs_300.h5')
    # # Prediksi menggunakan model yang sudah di-load
    # data_baru = np.array(data_baru)
    # y_pred_b = model.predict(data_baru)

    # # Konversi probabilitas menjadi kelas biner dengan threshold 0.5
    # threshold = 0.5
    # y_pred_class_b = (y_pred_b >= threshold).astype(int)
    # if y_pred_class_b == 0:
    #     st.write('Komplikasi')
    # elif y_pred_class_b == 1:
    #     st.write('Tidak Komplikasi')

