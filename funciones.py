import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm  # Importa la clase tqdm para la barra de progreso

#Función para la matriz de correlación de Pearson
def matriz_corr(df):
    plt.figure(figsize=(12, 8))  # Modifica los valores de ancho y alto según tus necesidades
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues', annot_kws={"size": 8})
    plt.title("Matriz de correlación")

#Función para realizar gráficos de línea
def graf_linea(df, fecha, valores_y1, *valores_y):
    plt.figure(figsize=(10, 4), facecolor='white')
    sns.set(style="darkgrid")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", '#9a2ca0'
    ,'#000000','#7f7f7f','#bcbd22']

    valores = [valores_y1, *valores_y]

    for i, valor in enumerate(valores):
        plt.plot(df[fecha], df[valor], label=valor, color=colors[i])

    plt.title('Evolución de ' + ', '.join(valores))
    plt.legend(frameon=False)
    plt.xticks(rotation=90)

    # Configurar los ticks del eje x cada 5 años
    # plt.xticks(range(df[fecha].min(), df[fecha].max(), 5))

    plt.show()

#Función para gráficos de barras
def graf_barras(df,valores_x,valores_y):
    df = df.sort_values(valores_y, ascending=False)
    df.plot(kind='bar', x=valores_x, y=valores_y, color='#08306B', width=0.9,figsize=(10,5))
    plt.title(valores_y + ' vs ' + valores_x)
    plt.show()

#Función para gráficos de dispersión
def graf_dispersion(df, valores_x, valores_y):
    # Filtrar los valores NaN en las columnas de interés
    df_filtered = df[[valores_x, valores_y]].dropna()

    fig, ax = plt.subplots()
    df_filtered.plot(x=valores_x, y=valores_y, kind='scatter', ax=ax)
    ax.set_xlabel(valores_x)
    ax.set_ylabel(valores_y)
    ax.set_title('Gráfico de dispersión entre ' + valores_x + ' y ' + valores_y)
    plt.show()


# Función para gráficos de dispersión comparando entre series
def graf_dispersion_comparativa(df, locations, x_var, y_var, marker_size=10):
    plt.figure(figsize=(10, 3))  # Define el tamaño de la figura

    # Bucle para cada localización
    for loc in locations:
        # Filtrar los datos
        subset = df[df['Location'] == loc]
        
        # Convertir la columna de fecha si es necesario
        if pd.api.types.is_string_dtype(subset[x_var]):
            subset[x_var] = pd.to_datetime(subset[x_var])

        # Graficar
        plt.scatter(subset[x_var], subset[y_var], label=loc, s=marker_size)

    # Decorar el gráfico
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'Dispersión de {y_var} en diferentes localizaciones')
    plt.legend()
    plt.show()

#Función para gráficos de bigotes, 0 para un diseño compacto, 1 para un diseño más detallado
def bigote(df,columna,diseño=0):
    if diseño==0:
      plt.figure(figsize=(8, 0.2))  # Ajusta el ancho y la altura según tus preferencias
      sns.boxplot(data=df, x=columna, orient='h', color='#08306B')
      plt.xlabel(columna)
      plt.show()
    elif diseño==1:
      Q1 = df[columna].quantile(q=0.25)
      Q3 = df[columna].quantile(q=0.75)
      IQR = Q3 - Q1
      fig = plt.figure(figsize =(10, 7))
      plt.boxplot(df[columna], patch_artist=True, boxprops=dict(facecolor='#08306B'))
      plt.text(1.1, df[columna].median(), "Mediana: {:.1f}".format(df[columna].median()),fontsize=8)
      plt.text(1.1, df[columna].quantile(q=0.25), "Q1: {:.1f}".format(df[columna].quantile(q=0.25)),fontsize=8)
      plt.text(1.1, df[columna].quantile(q=0.75), "Q3: {:.1f}".format(df[columna].quantile(q=0.75)),fontsize=8)
      plt.text(0.85, df[columna].median(), "IQR: {:.1f}".format(IQR),fontsize=8)
      plt.xlabel(columna)

#Función para explorar las medidas principales de un df y su correlación
def explorar(df):
    # Filtrar las columnas numéricas
    columnas_numericas = df.select_dtypes(include='number').columns
    # Iterar sobre cada columna numérica y llamar a la función bigote
    for columna in columnas_numericas:
        histograma(df,columna)
    matriz_corr(df)

#Función para ver NaN de un df y su posibilidad de tratamiento
def analisis_nan(df):
    # Calcula la matriz de correlación
    correlation_matrix = df.corr(numeric_only=True)

    # Crea el nuevo DataFrame df_nan
    df_nan = pd.DataFrame(columns=['Columna', 'Max_Abs_Correlacion', 'Valor_Correlacion', 'Valores_NaN'])

    # Itera sobre las columnas y llena df_nan con los valores correspondientes
    for columna in correlation_matrix.columns:
        max_corr_columna = correlation_matrix[columna].drop(columna).abs().idxmax()
        max_corr_valor = correlation_matrix[columna][max_corr_columna]
        valores_nan = df[columna].isna().sum()
        df_nan = pd.concat([df_nan, pd.DataFrame({'Columna': [columna],
                                                  'Max_Abs_Correlacion': [max_corr_columna],
                                                  'Valor_Correlacion': [max_corr_valor],
                                                  'Valores_NaN': [valores_nan]})], ignore_index=True)

    return df_nan

def recrear_nan(df, columna_nan, columna_correlacionada):
    # Crear un DataFrame sin los valores NaN de las columnas involucradas
    df_no_nan = df.dropna(subset=[columna_nan, columna_correlacionada])

    # Crear un modelo de regresión lineal
    model = LinearRegression()

    # Ajustar el modelo con los datos existentes
    model.fit(df_no_nan[columna_correlacionada].values.reshape(-1, 1), df_no_nan[columna_nan].values.reshape(-1, 1))

    # Obtener los valores NaN de la variable columna_nan
    # Asegura además que no haya NaN en el predictor
    df_nan = df[df[columna_nan].isnull() & df[columna_correlacionada].notnull()]

    # Predecir los valores NaN utilizando el modelo de regresión lineal
    predicted_values = model.predict(df_nan[columna_correlacionada].values.reshape(-1, 1))

    # Asignar los valores predichos a los valores NaN en la variable columna_nan usando los índices de df_nan
    df.loc[df_nan.index, columna_nan] = predicted_values.flatten()

def comparar_datos(df, ciudad1, ciudad2, variable):
    # Filtrar los datos para la primera ciudad donde la variable no es NaN.
    ciudad1_data = df[(df['Location'] == ciudad1) & df[variable].notna()]

    # Extraer las fechas donde la primera ciudad tiene datos de la variable
    ciudad1_dates = ciudad1_data['Date']

    # Filtrar los datos para la segunda ciudad en las mismas fechas.
    ciudad2_data = df[(df['Location'] == ciudad2) & df['Date'].isin(ciudad1_dates)]

    # Unir los datos de ambas ciudades basándonos en las fechas.
    comparison_data = pd.merge(ciudad1_data, ciudad2_data, on='Date', suffixes=(f'_{ciudad1}', f'_{ciudad2}'))

    # Calcular la correlación para la variable especificada
    correlation = comparison_data[f'{variable}_{ciudad1}'].corr(comparison_data[f'{variable}_{ciudad2}'])

    # Imprimir resultados
    print(f"Correlación entre {ciudad1} y {ciudad2} para {variable}: {correlation}")

def recrear_geo_nan(df, df_data, loc_nan, loc_data, var):
    """
    Reemplaza los valores NaN para una variable específica en una localidad, utilizando los valores
    de la misma variable en otra localidad para el mismo día, y cuenta cuántos días únicos fueron usados para reemplazar esos NaN.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos de la localidad con valores NaN.
    df_data (DataFrame): El DataFrame que contiene los datos de la localidad de donde se tomarán los valores para reemplazar.
    loc_nan (str): Nombre de la localidad que tiene valores NaN que deben ser reemplazados.
    loc_data (str): Nombre de la localidad de donde se tomarán los valores para reemplazar.
    var (str): Nombre de la variable para la cual se reemplazarán los NaN.

    Retorna:
    DataFrame: DataFrame con los valores NaN reemplazados donde fue posible.
    """
    # Crear copia del DataFrame para evitar modificar el original
    df_copy = df.copy()

    # Datos de la localidad con datos disponibles en df_data
    data_values = df_data[(df_data['Location'] == loc_data) & df_data[var].notnull()][['Date', var]]

    # Renombrar la columna de valores para evitar conflictos durante el merge
    data_values.rename(columns={var: f'{var}_replacement'}, inplace=True)

    # Datos de la localidad con NaNs en df
    nan_values = df_copy[(df_copy['Location'] == loc_nan) & df_copy[var].isnull()][['Date']]

    # Merge left para combinar los datos de df con los valores de df_data
    df_merged = pd.merge(df_copy, data_values, on='Date', how='left')

    # Condición para reemplazar NaNs
    condition = (df_merged['Location'] == loc_nan) & (df_merged[var].isnull())

    # Contar días únicos donde se reemplazan valores
    days_used_for_replacement = pd.merge(nan_values, data_values, on='Date', how='inner')['Date'].nunique()

    # Reemplazo de los NaN
    df_merged.loc[condition, var] = df_merged.loc[condition, f'{var}_replacement']

    # Eliminar la columna de reemplazo
    df_merged.drop(columns=[f'{var}_replacement'], inplace=True)

    print(f"La cantidad de datos reemplazados fue {days_used_for_replacement}")
    return df_merged

def analisis_geo_nan(df, var, dist_matrix, cities_coords):
    """
    Analiza la distribución de valores NaN para una variable específica en distintas ciudades
    y encuentra la ciudad más cercana para cada una de ellas junto con la distancia.

    Parámetros:
    df (DataFrame): DataFrame de donde se extraerán los datos.
    var (str): Variable para la cual se contarán los NaNs.
    dist_matrix (DataFrame): Matriz de distancias entre ciudades.
    cities_coords (dict): Diccionario de coordenadas de las ciudades.

    Retorna:
    DataFrame: Información combinada que incluye conteo de NaNs, ciudad más cercana y distancia.
    """
    # DataFrame para almacenar la ciudad más cercana y la distancia
    closest_city_and_distance = pd.DataFrame(index=cities_coords.keys(), columns=['Nearest City', 'Distance (km)'])

    # Encontrar la ciudad más cercana y la distancia para cada una
    for city in dist_matrix.columns:
        closest_city = dist_matrix[city].drop(city).idxmin()
        closest_distance = dist_matrix.at[city, closest_city]
        closest_city_and_distance.at[city, 'Nearest City'] = closest_city
        closest_city_and_distance.at[city, 'Distance (km)'] = closest_distance

    # Contar NaN en la variable específica y agrupar por ciudad
    nan_count_per_city = df[df[var].isna()]['Location'].value_counts()

    # Convertir la serie en un DataFrame
    nan_count_df = nan_count_per_city.reset_index()
    nan_count_df.columns = ['Location', f'{var} NaN Count']

    # Unir nan_count_df con closest_city_and_distance, manteniendo nan_count_df como el DataFrame principal
    complete_info = nan_count_df.merge(closest_city_and_distance, how='left', left_on='Location', right_index=True)

    return complete_info

def procesar_geo_nan(df, df_data, dist_matrix, cities_coords, distance_threshold=100):
    """
    Procesa los NaN en 'df' utilizando los valores de 'df_data' basándose en la proximidad geográfica definida
    por 'dist_matrix' y 'cities_coords', con un umbral de distancia especificado.

    Parámetros:
    df (DataFrame): DataFrame con datos que incluyen NaNs que necesitan ser procesados.
    df_data (DataFrame): DataFrame de donde se tomarán los datos para reemplazar los NaNs.
    dist_matrix (DataFrame): DataFrame que contiene las distancias entre localidades.
    cities_coords (dict): Diccionario con las coordenadas de las localidades.
    distance_threshold (int, opcional): Distancia máxima en km para considerar una ciudad cercana.

    Retorna:
    DataFrame: DataFrame con los valores NaN procesados.
    """
    excluded_columns = ['Date', 'Location','RainTomorrow','RainfallTomorrow']
    included_columns = [col for col in df.columns if col not in excluded_columns]

    print("Procesando las siguientes columnas:", included_columns)
    print("El rango fijado para ciudades cercanas es (km):", distance_threshold)
    for variable in included_columns:
        print(f"Analizando la variable: {variable}")
        geo_analysis = analisis_geo_nan(df, variable, dist_matrix, cities_coords)
        print("Resultado del análisis geográfico:\n", geo_analysis)

        for index, row in geo_analysis.iterrows():
            location = row['Location']
            if pd.isnull(row['Distance (km)']):
                continue
            # Ordenar ciudades por distancia, manteniendo solo las dentro del umbral
            possible_cities = dist_matrix[location].sort_values()
            possible_cities = possible_cities[(possible_cities > 0) & (possible_cities <= distance_threshold)]

            for closest_city, distance in possible_cities.items():
                if pd.notnull(distance):
                    print(f"Procesando la localidad: {location}")
                    print(f"Intentando con la ciudad más cercana {closest_city} a {distance} km.")
                    df = recrear_geo_nan(df, df_data, location, closest_city, variable)
                    if not df[df['Location'] == location][variable].isna().any():
                        print(f"Todos los NaNs fueron reemplazados para {location} en la variable {variable}.")
                        break
                else:
                    print(f"No se encontró ciudad cercana adecuada para {location}")

    remaining_nans = df.isna().sum().sort_values(ascending=False)
    print("NaN restantes después del procesamiento:\n", remaining_nans)

    # Crear una copia del DataFrame sin las columnas 'RainfallTomorrow' y 'RainTomorrow'
    df_filtered = df.drop(columns=excluded_columns)

    # Contar filas que contienen al menos un valor NaN en el DataFrame filtrado
    nan_rows_count_filtered = df_filtered.isna().any(axis=1).sum()
    print(f'Número de filas con al menos un valor NaN: {nan_rows_count_filtered}')
    
    return df

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def gradient_descent(X_train, y_train, X_test, y_test, lr=0.01, epochs=100):
    """
    shapes:
        X_train = nxm
        y_train = nx1
        X_test = pxm
        y_test = px1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]
    
    o = X_test.shape[0]

    # Poner columna de unos a las matrices X
    X_train = np.hstack((np.ones((n, 1)), X_train))
    X_test = np.hstack((np.ones((o, 1)), X_test))
    

    # Inicializar pesos aleatorios
    W = np.random.randn(m+1).reshape(m+1, 1)

    train_errors = []  # Para almacenar el error de entrenamiento en cada época
    test_errors = []   # Para almacenar el error de prueba en cada época

    for i in range(epochs):
        # Calcular predicción y error de entrenamiento
        prediction_train = np.matmul(X_train, W) 
        error_train = y_train - prediction_train  
        train_rmse = np.sqrt(np.mean(error_train ** 2))  # Calcular RMSE
        train_errors.append(train_rmse)

        # Calcular predicción y error de prueba
        prediction_test = np.matmul(X_test, W) 
        error_test = y_test - prediction_test 
        test_rmse = np.sqrt(np.mean(error_test ** 2))  # Calcular RMSE
        test_errors.append(test_rmse)

        # Calcular el gradiente y actualizar pesos
        grad_sum = np.sum(error_train * X_train, axis=0)
        grad_mul = -2/n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

        W = W - (lr * gradient)

    # Calcular las predicciones finales para entrenamiento y prueba
    prediction_train_final = np.matmul(X_train, W)
    prediction_test_final = np.matmul(X_test, W)

    # Calcular RMSE final para entrenamiento y prueba
    final_train_rmse = np.sqrt(mean_squared_error(y_train, prediction_train_final))
    final_test_rmse = np.sqrt(mean_squared_error(y_test, prediction_test_final))

    # Calcular R^2 final para entrenamiento y prueba
    final_train_r2 = r2_score(y_train, prediction_train_final)
    final_test_r2 = r2_score(y_test, prediction_test_final)

    # Graficar errores de entrenamiento y prueba
    plt.figure(figsize=(12, 6))
    plt.plot(train_errors, label='Error de entrenamiento')
    plt.plot(test_errors, label='Error de test')
    plt.xlabel('Época')
    plt.ylabel('Error RMSE')
    plt.legend()
    plt.title('RMSE de entrenamiento y prueba vs iteraciones (GD)')
    plt.show()

    return {'test_rmse': final_test_rmse, 'test_r2': final_test_r2, 'train_rmse': final_train_rmse, 'train_r2': final_train_r2, 'predictores': W}

from tqdm import tqdm  # Importa la clase tqdm para la barra de progreso

def stochastic_gradient_descent(X_train, y_train, X_test, y_test, lr=0.01, epochs=100):
    n = X_train.shape[0]
    m = X_train.shape[1]

    X_train = np.hstack((np.ones((n, 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    W = np.random.randn(m + 1).reshape(-1, 1)

    train_errors = []
    test_errors = []

    # Agregamos tqdm al bucle de épocas para monitorear el progreso
    for epoch in tqdm(range(epochs), desc='Training Epochs', unit='epoch'):
        # Permutación aleatoria de los datos
        permutation = np.random.permutation(n)
        X_train_permuted = X_train[permutation]
        y_train_permuted = y_train[permutation]

        # Uso de tqdm en el bucle interno para ver el progreso de cada iteración de muestra
        for j in tqdm(range(n), desc=f'Epoch {epoch+1}', leave=False):
            x_sample = X_train_permuted[j]
            y_sample = y_train_permuted[j]

            prediction = np.matmul(x_sample, W)
            error_train = y_sample - prediction
            train_rmse = np.sqrt(np.mean(error_train ** 2))  # Calcular RMSE
            train_errors.append(train_rmse)

            gradient = -2 * error_train * x_sample.reshape(-1, 1)
            W = W - (lr * gradient)

            prediction_test = np.matmul(X_test, W)
            error_test = y_test - prediction_test
            test_rmse = np.sqrt(np.mean(error_test ** 2))
            test_errors.append(test_rmse)

    # Calcular las predicciones finales para entrenamiento y prueba
    prediction_train_final = np.matmul(X_train, W)
    prediction_test_final = np.matmul(X_test, W)

    # Calcular RMSE final para entrenamiento y prueba
    final_train_rmse = np.sqrt(mean_squared_error(y_train, prediction_train_final))
    final_test_rmse = np.sqrt(mean_squared_error(y_test, prediction_test_final))

    # Calcular R^2 final para entrenamiento y prueba
    final_train_r2 = r2_score(y_train, prediction_train_final)
    final_test_r2 = r2_score(y_test, prediction_test_final)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_errors, label='Error de entrenamiento')
    plt.plot(test_errors, label='Error de prueba')
    plt.xlabel('Iteración')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE de entrenamiento y prueba vs iteraciones (SGD)')
    plt.show()

    return {'test_rmse': final_test_rmse, 'test_r2': final_test_r2, 'train_rmse': final_train_rmse, 'train_r2': final_train_r2, 'predictores': W}

def mini_batch_gradient_descent(X_train, y_train, X_test, y_test, lr=0.01, epochs=100, batch_size=11):
    n = X_train.shape[0]
    m = X_train.shape[1]

    X_train = np.hstack((np.ones((n, 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    W = np.random.randn(m + 1).reshape(-1, 1)

    train_errors = []
    test_errors = []

    # Barra de progreso para las épocas
    for i in tqdm(range(epochs), desc='Training Epochs', unit='epoch'):
        # Permutación aleatoria de los datos
        permutation = np.random.permutation(n)
        X_train_permuted = X_train[permutation]
        y_train_permuted = y_train[permutation]

        # Barra de progreso para los mini-lotes dentro de cada época
        batch_iter = range(0, n, batch_size)
        for j in tqdm(batch_iter, desc=f'Epoch {i+1}', leave=False):
            # Asegurarse de no sobrepasar el tamaño de la muestra
            end_index = min(j + batch_size, n)
            x_batch = X_train_permuted[j:end_index, :]
            y_batch = y_train_permuted[j:end_index].reshape(-1, 1)

            prediction = np.matmul(x_batch, W)
            error_train = y_batch - prediction
            train_rmse = np.sqrt(np.mean(error_train ** 2))  # Calcular RMSE
            train_errors.append(train_rmse)

            gradient = -2 * np.matmul(x_batch.T, error_train) / x_batch.shape[0]  # Usamos x_batch.shape[0] para lidiar con el último lote

            W = W - (lr * gradient)

            prediction_test = np.matmul(X_test, W)
            error_test = y_test - prediction_test
            test_rmse = np.sqrt(np.mean(error_test ** 2))
            test_errors.append(test_rmse)

    # Calcular las predicciones finales para entrenamiento y prueba
    prediction_train_final = np.matmul(X_train, W)
    prediction_test_final = np.matmul(X_test, W)

    # Calcular RMSE final para entrenamiento y prueba
    final_train_rmse = np.sqrt(mean_squared_error(y_train, prediction_train_final))
    final_test_rmse = np.sqrt(mean_squared_error(y_test, prediction_test_final))

    # Calcular R^2 final para entrenamiento y prueba
    final_train_r2 = r2_score(y_train, prediction_train_final)
    final_test_r2 = r2_score(y_test, prediction_test_final)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_errors, label='Error de entrenamiento')
    plt.plot(test_errors, label='Error de prueba')
    plt.xlabel('Iteración')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE de entrenamiento y prueba vs iteraciones (Mini-Batch GD)')
    plt.show()

    return {'test_rmse': final_test_rmse, 'test_r2': final_test_r2, 'train_rmse': final_train_rmse, 'train_r2': final_train_r2, 'predictores': W}

import seaborn as sns
import matplotlib.pyplot as plt

def histograma(data, columna, estilo='darkgrid', facecolor='#1E1E1E', gridcolor='#2C2F33', textcolor='white', edgecolor='gray'):
    # Configuración del estilo de los gráficos
    sns.set(style=estilo)
    plt.rcParams['axes.facecolor'] = facecolor
    plt.rcParams['figure.facecolor'] = facecolor
    plt.rcParams['grid.color'] = gridcolor
    plt.rcParams['text.color'] = textcolor
    plt.rcParams['axes.labelcolor'] = textcolor
    plt.rcParams['xtick.color'] = textcolor
    plt.rcParams['ytick.color'] = textcolor
    plt.rcParams['axes.edgecolor'] = edgecolor

    # Crear un displot con un mapa de densidad de kernel (KDE)
    g = sns.displot(x=columna, data=data, palette='Set2', kind='kde', height=5, aspect=1.5, color="#007ACC")

    # Calcular estadísticas relevantes
    min_val = data[columna].min()
    max_val = data[columna].max()
    mean_val = data[columna].mean()
    std_dev = data[columna].std()
    cuartiles = data[columna].quantile([0.25, 0.5, 0.75])
    iqr = cuartiles[0.75] - cuartiles[0.25]
    lower_bound_iqr = cuartiles[0.25] - 1.5 * iqr
    upper_bound_iqr = cuartiles[0.75] + 1.5 * iqr

    # 3σ limits
    lower_bound_3std = mean_val - 3 * std_dev
    upper_bound_3std = mean_val + 3 * std_dev

    # Select more permissive bounds
    lower_bound = lower_bound_3std
    upper_bound = upper_bound_3std

    # Determine the method used
    method_used = "IQR" if (lower_bound == lower_bound_iqr and upper_bound == upper_bound_iqr) else "3σ"

    # Plot lines for the bounds
    plt.axvline(x=lower_bound, color='salmon', linestyle='--', linewidth=2)
    plt.axvline(x=upper_bound, color='salmon', linestyle='--', linewidth=2)
    plt.text(lower_bound, plt.gca().get_ylim()[1], f'Lower Bound ({method_used})', color='salmon', ha='right', va='top', rotation=90)
    plt.text(upper_bound, plt.gca().get_ylim()[1], f'Upper Bound ({method_used})', color='salmon', ha='left', va='top', rotation=90)

    # Plot line for the mean
    plt.axvline(x=mean_val, color="#007ACC", linestyle='-', linewidth=2)
    plt.text(mean_val + 0.1 * std_dev, plt.gca().get_ylim()[1] / 2, f'Mean: {mean_val:.2f}', color="#007ACC", ha='left', rotation=90)

    # Título del gráfico elevado para evitar superposición
    plt.gcf().suptitle(f'Histograma de {columna}', color=textcolor, y=1.05)  # Elevamos y ajustamos el color del título

    # Agregar líneas verticales y texto para los cuartiles con etiquetas Q1, Q2, Q3
    quartile_labels = ['Q1', 'Q2', 'Q3']
    for q, label in zip(cuartiles.index, quartile_labels):
        value = cuartiles[q]
        plt.axvline(x=value, color='#ffffff', linestyle='-', linewidth=1)
        plt.text(value, plt.gca().get_ylim()[1]*1.05, f'{label}', color='#ffffff', ha='center', va='top', rotation=0)

    # Añadir un cuadro de texto para resumir estadísticas
    stats_text = (f'Mean: {mean_val:.2f}\n'
                  f'Std Dev: {std_dev:.2f}\n'
                  f'Min: {min_val:.2f}\n'
                  f'Max: {max_val:.2f}\n'
                  f'Q1\n'
                  f'Q2 (Median)\n'
                  f'Q3\n'
                  f'Lower Bound ({method_used}): {lower_bound:.2f}\n'
                  f'Upper Bound ({method_used}): {upper_bound:.2f}')
    plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, color=textcolor, ha='left', va='center')

    # Muestra el gráfico
    plt.show()