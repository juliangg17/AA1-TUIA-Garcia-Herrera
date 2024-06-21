import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from funciones import *
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import os
import warnings
warnings.filterwarnings("ignore")

# DATA EXTRACTION
model_save_dir  = 'G:/Mi unidad/TUIA/IA4.1 AA I/Trabajo Práctico Integrador/AA1-TUIA-Garcia-Herrera/'
data_path = 'G:/Mi unidad/TUIA/IA4.1 AA I/Trabajo Práctico Integrador/AA1-TUIA-Garcia-Herrera/weatherAUS.csv'
df = pd.read_csv(data_path, sep=',',engine='python')
#DATA SPLIT
df_train, _ = data_split(df)
#DATA PREPARATION
df_train, df_49 = source_data_preparation(df_train)
df_train=impute(df_train,df_49)

# ENTRENAMIENTO NN PARA CLASIFICACIÓN
X_train, y_train = split_x_y(df_train,'clasificacion')
set_seed(50)
pipeline_clas = Pipeline([
    ('oversampler',RandomOverSampler(random_state=42)),
    ('scaler', StandardScaler()),
    ('nn_classifier', NNClassifier(build_fn=nn_clas))
])
pipeline_clas.fit(X_train, y_train)
model_clas = pipeline_clas.named_steps['nn_classifier'].get_model()
save_model(model_clas, os.path.join(model_save_dir, 'keras_model_clas.h5'))
pipeline_clas.named_steps['nn_classifier'].set_model(None)
joblib.dump(pipeline_clas, os.path.join(model_save_dir, 'pipeline_clas.joblib'))

# ENTRENAMIENTO NN PARA REGRESIÓN
X_train, y_train = split_x_y(df_train,'regresion')
pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('nn_regressor', NNRegressor(build_fn=nn_reg))
])
pipeline_reg.fit(X_train, y_train)
model_reg = pipeline_reg.named_steps['nn_regressor'].get_model()
save_model(model_reg, os.path.join(model_save_dir, 'keras_model_reg.h5'))
pipeline_reg.named_steps['nn_regressor'].set_model(None)
joblib.dump(pipeline_reg, os.path.join(model_save_dir, 'pipeline_reg.joblib'))