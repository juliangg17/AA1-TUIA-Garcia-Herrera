import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#Función para la matriz de correlación de Pearson
def matriz_corr(df):
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues', annot_kws={"size": 6})
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
        bigote(df,columna)
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

def recrear_geo_nan(df, loc_nan, loc_data, var):
    """
    Reemplaza los valores NaN para una variable específica en una localidad, utilizando los valores
    de la misma variable en otra localidad para el mismo día, y cuenta cuántos días únicos fueron usados para reemplazar esos NaN.

    Parámetros:
    df (DataFrame): El DataFrame que contiene los datos.
    loc_nan (str): Nombre de la localidad que tiene valores NaN que deben ser reemplazados.
    loc_data (str): Nombre de la localidad de donde se tomarán los valores para reemplazar.
    var (str): Nombre de la variable para la cual se reemplazarán los NaN.

    Retorna:
    DataFrame: DataFrame con los valores NaN reemplazados donde fue posible.
    """
    # Crear copia del DataFrame para evitar modificar el original
    df_copy = df.copy()

    # Datos de la localidad con datos disponibles
    data_values = df_copy[(df_copy['Location'] == loc_data) & df_copy[var].notnull()][['Date', var]]

    # Renombrar la columna de valores para evitar conflictos durante el merge
    data_values.rename(columns={var: f'{var}_replacement'}, inplace=True)

    # Datos de la localidad con NaNs
    nan_values = df_copy[(df_copy['Location'] == loc_nan) & df_copy[var].isnull()][['Date']]

    # Merge left para combinar los datos
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

def procesar_geo_nan(df, dist_matrix, cities_coords, distance_threshold=100):
    excluded_columns = ['Date', 'Location']
    non_boolean_columns = df.select_dtypes(exclude=[bool]).columns.difference(excluded_columns)

    print("Procesando las siguientes columnas:", non_boolean_columns)
    print("El rango fijado para ciudades cercanas es (km):",distance_threshold)
    for variable in non_boolean_columns:
        print(f"Analizando la variable: {variable}")
        geo_analysis = analisis_geo_nan(df, variable, dist_matrix, cities_coords)
        print("Resultado del análisis geográfico:\n", geo_analysis)

        for index, row in geo_analysis.iterrows():
            location = row['Location']
            closest_city = row['Nearest City']
            distance = row['Distance (km)']

            if pd.notnull(distance) and 0 < float(distance) < distance_threshold:
                print(f"Procesando la localidad: {location}")
                print(f"La ciudad más cercana a {location} es {closest_city} a {distance} km.")
                df = recrear_geo_nan(df, location, closest_city, variable)
            else:
                print(f"No se encontró ciudad cercana en el rango deseado para {location}")

    remaining_nans = df.isna().sum().sort_values(ascending=False)
    print("NaN restantes después del procesamiento:\n", remaining_nans)
    return df