import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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