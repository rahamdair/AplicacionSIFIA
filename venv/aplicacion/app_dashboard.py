import dash
from dash import dcc, html, Output, Input, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve


import joblib
import base64
import io
from dash.dash_table import DataTable

# Se carga el dataset
data = pd.read_csv('venv/aplicacion/datasets/Churn_Modelling_clean_traduccion.csv')

# Se inicia la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Se define rangos de edades para graficar
bins = [18, 25, 35, 44, 55, 65, 75, 85]
etiquetas = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=etiquetas, right=False)

# ------------------------------------------------------------------------------
# Cargar el dataset para la generación del modelo
data_esc1 = pd.read_csv("venv/aplicacion/datasets/Churn_Modelling_escenario1.csv")

# División del dataset en entrenamiento y prueba
features = data_esc1.drop("Exited", axis=1)
target = data_esc1["Exited"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Se realiza el proceso de estandarización de las características (features)
escalado = StandardScaler()
# Se estandariza el conjunto de datos original (desbalanceado)
X_train_std = escalado.fit_transform(X_train)
X_test_std = escalado.transform(X_test)
# ------------------------------------------------------------------------------

# Layout de la aplicación
app.layout = html.Div([
    # --- Rutas del Dashboard
    html.Div([
        html.H1('SIF-IA'),
        dcc.Link('Datos de Clientes', href='/dashboard1', style={'margin': '100px', 'color': 'white', 'fontSize': '24px'}),
        dcc.Link('Análisis', href='/dashboard2', style={'margin': '100px', 'color': 'white','fontSize': '24px'}),
        dcc.Link('Evaluación de Modelos', href='/dashboard3', style={'margin': '100px', 'color': 'white', 'fontSize': '24px'}),
        dcc.Link('Modelo de Predicción', href='/dashboard4', style={'margin': '100px', 'color': 'white', 'fontSize': '24px'}),
        html.Img(src='assets/logoUTPL.png')
    ], className='banner'),

    dcc.Location(id='url', refresh=False),
    html.Div(id='contenido-pagina')
])

# Callback para el contenido de las rutas
@app.callback(
    Output('contenido-pagina', 'children'),
    Input('url', 'pathname'))

def display_pagina(pathname):
    if pathname == '/dashboard1':
        return dashboard1_layout()
    elif pathname == '/dashboard2':
        return dashboard2_layout()
    elif pathname == '/dashboard3':
        return dashboard3_layout()
    elif pathname == '/dashboard4':
        return dashboard4_layout()
    else:
        return dashboard1_layout()


def dashboard1_layout():
    return html.Div([
        # --- Contenedores de Información General
        html.Div([
            # Contenedor 1: Suma Total de Balance
            html.Div([
                html.H3('Balance Total'),
                html.P(id='balance-total', style={'fontSize': '22px'})
            ], className='create_container2'),
            # Contenedor 2: Salario Estimado Total
            html.Div([
                html.H3('Salario Estimado Total'),
                html.P(id='salario-total', style={'fontSize': '22px'})
            ], className = 'create_container2'),
            # Contenedor 3: Puntaje Crediticio Promedio
            html.Div([
                html.H3('Puntaje Crediticio Promedio'),
                html.P(id='clientes-total', style={'fontSize': '22px'})
            ], className = 'create_container2'),
            # Contenedor 4: Información del Dataset
            html.Div([
                html.H3('Información del Dataset'),
                #html.H4('Fuente: Kaggle'),
                html.P('Modelo de abandono de clientes', style={'fontSize': '22px'})
            ], className = 'create_container2')
        ], className = 'row flex-display'),  

        # ---- Filtros de Selección
        html.Div([
            # Contenedor: Selección de Geografía
            html.Div([
                #html.H4('Selección por Geografía'),
                dcc.Dropdown(
                    id='geografia-dropdown',
                    options=[{'label': geo, 'value': geo} for geo in data['Geography'].unique()],
                    multi=True,
                    placeholder="Selecciona la geografía",
                    style={'fontSize': 20}
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),

            # Contenedor: Selección de Género
            html.Div([
                #html.H4('Selección por Género'),
                dcc.Dropdown(
                    id='genero-dropdown',
                    options=[{'label': gen, 'value': gen} for gen in data['Gender'].unique()],
                    multi=True,
                    placeholder="Selecciona el género",
                    style={'fontSize': 20}
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'})
        ], className = 'row flex-display'),

        # --- Gráficas primer nivel
        html.Div([
            # Gráfica de pie de Geografía
            html.Div([
                dcc.Graph(id='geografia-pie-chart')
            ], className='create_container2'),
            # Gráfica de pie (dona) de género
            html.Div([
                dcc.Graph(id='genero-dona-chart')
            ], className='create_container2'),
            # Gráfica de barras para rangos de edades
            html.Div([
                dcc.Graph(id='edad-bar-chart')
            ], className='create_container2')
        ], className = 'row flex-display'),

        # --- Gráficas segundo nivel

        html.Div([
            html.Div([
                #dcc.Graph(id='puntajecrediticio-indicador-chart')
                dcc.Graph(id='exited-barra-chart')
            ], className='create_container2'),
            html.Div([
                dcc.Graph(id='balance-linea-chart')
            ], className='create_container2'),
            html.Div([
                dcc.Graph(id='salario-linea-chart')
            ], className='create_container2')
        ], className='row flex-display')
    ])


# Callback para Dashboard 1
@app.callback(
    Output(component_id='balance-total', component_property='children'),
    Output(component_id='salario-total', component_property='children'),
    Output(component_id='clientes-total', component_property='children'),
    Output(component_id='geografia-pie-chart', component_property='figure'),
    Output(component_id='genero-dona-chart', component_property='figure'),
    Output(component_id='edad-bar-chart', component_property='figure'),
    Output(component_id='balance-linea-chart', component_property='figure'),
    Output(component_id='salario-linea-chart', component_property='figure'),
    Output(component_id='exited-barra-chart', component_property='figure'),
    Input(component_id='geografia-dropdown', component_property='value'),
    Input(component_id='genero-dropdown', component_property='value')
)


def cargar_dashboard1(seleccion_geografia, seleccion_genero):
    # Filtrado de datos
    filtrado_data = data.copy()

    if seleccion_genero:
        filtrado_data = filtrado_data[filtrado_data['Gender'].isin(seleccion_genero)]

    if seleccion_geografia:
        filtrado_data = filtrado_data[filtrado_data['Geography'].isin(seleccion_geografia)]

    # Calcular totales para balance y salario
    balance_total = filtrado_data['Balance'].sum()
    salario_total = filtrado_data['EstimatedSalary'].sum()
    clientes_total = len(filtrado_data)

    # Gráfico de pie por Geografía
    geo_cuenta = filtrado_data['Geography'].value_counts().reset_index()
    geo_cuenta.columns = ['Geography', 'Count']
    geo_pie_fig = px.pie(geo_cuenta, values='Count', names='Geography', 
                         title='Distribución de Cientes por Geografía')
                         #color_discrete_sequence=['navy', 'green', 'orange'])
    # Personalizar el texto de porcentaje
    #geo_pie_fig.update_traces(textinfo='percent+label', textfont_size=16)
    geo_pie_fig.update_traces(textinfo='percent+label', textfont=dict(size=16, color='white'))
    # Centrar el título de la gráfica
    geo_pie_fig.update_layout(title_x=0.5, title_font=dict(size=20))


    # Gráfico de dona por Género
    genero_cuenta = filtrado_data['Gender'].value_counts().reset_index()
    genero_cuenta.columns = ['Gender', 'Count']
    genero_dona_fig = px.pie(genero_cuenta, values='Count', names='Gender', 
                             title='Distribución de Clientes por Género', hole=0.3)
    # Personalizar el texto de porcentaje
    #genero_dona_fig.update_traces(textinfo='percent+label', textfont_size=16)
    genero_dona_fig.update_traces(textinfo='percent+label', textfont=dict(size=16, color='white'))

    # Centrar el título de la gráfica
    genero_dona_fig.update_layout(title_x=0.5, title_font=dict(size=20))

    # Contar el número de clientes en cada grupo de edad
    edad_cuenta = filtrado_data['Age_Group'].value_counts().reset_index()
    edad_cuenta.columns = ['Age_Group', 'Count']
    edad_cuenta = edad_cuenta.sort_values('Age_Group')  # Ordenar por grupo de edad (ascendente)
    edad_barras_fig = px.bar(edad_cuenta, x='Age_Group', y='Count', 
                             title='Distribución de Clientes por Edades',
                             labels={'Age_Group':'Grupo de Edad', 'Count':'Cantidad de Clientes'}, 
                             color_discrete_sequence=['navy'])
    # Centrar el título de la gráfica
    edad_barras_fig.update_layout(title_x=0.5, title_font=dict(size=20))

    # Agrupa por Tenure y calcula la suma total de Balance y EstimatedSalary
    grupo_balance = filtrado_data.groupby('Tenure')['Balance'].sum().reset_index()
    balance_linea_fig = px.line(
        grupo_balance,
        x='Tenure',
        y='Balance',
        title='Balance por el tiempo de Permanencia del Cliente',
        labels={'Tenure':'Permanencia (Años)', 'Balance':' Balance'},
        color_discrete_sequence=['navy'],
        markers=True)
    balance_linea_fig.update_layout(title_x=0.5, title_font=dict(size=20))
    

    grupo_salario = filtrado_data.groupby('Tenure')['EstimatedSalary'].sum().reset_index()
    salario_linea_fig = px.line(
        grupo_salario,
        x='Tenure',
        y='EstimatedSalary',
        title='Salario por el tiempo de Permanencia del Cliente',
        labels={'Tenure':'Permanencia (Años)', 'EstimatedSalary':'Salario Estimado'},
        color_discrete_sequence=['navy'],
        markers=True)
    salario_linea_fig.update_layout(title_x=0.5, title_font=dict(size=20))

    # Contar los valores de 'Exited'
    abandono_cuenta = filtrado_data['Exited'].value_counts().reset_index()
    abandono_cuenta.columns = ['Exited', 'Count']
    
    # Crear la gráfica de barras
    abandono_fig = px.bar(abandono_cuenta, x='Exited', y='Count', 
                        title='Distribución de Clientes por Abandono',
                        labels={'Exited': 'Abandono (0: No, 1: Si)', 'Count': 'Cantidad de Clientes'},
                        #color_discrete_map={0: 'blue', 1: 'orange'},
                        color='Exited',
                        color_continuous_scale=['tomato', 'blue']
                        )
    abandono_fig.update_layout(title_x=0.5, title_font=dict(size=20))

    return f'${balance_total:,.2f}', f'${salario_total:,.2f}', f'{clientes_total}', geo_pie_fig, genero_dona_fig, edad_barras_fig, balance_linea_fig, salario_linea_fig, abandono_fig



# --------Layout para Dashboard 2
def dashboard2_layout():
    return html.Div([

        html.Div([
            # Contenedor: Selección para gráfico de barras
            html.Div([
                dcc.Dropdown(
                    id='filtro-barras-dropdown',
                    options=[
                        {'label':'Número de Productos', 'value': 'NumOfProducts'},
                        {'label': 'Tiene Tarjeta de Crédito', 'value': 'HasCrCard'},
                        {'label': 'Miembro Activo', 'value': 'IsActiveMember'}
                    ],
                    value='NumOfProducts',  # Valor por defecto
                    style={'fontSize': 20}
                    #clearable=False
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),
            # Contenedor: Selección para gráfico de dispersión
            html.Div([
                dcc.Dropdown(
                    id='filtro-dispersion-dropdown',
                    options=[
                        {'label': 'Balance', 'value': 'Balance'},
                        {'label': 'Puntaje Crediticio', 'value':'CreditScore'},
                        {'label': 'Salario Estimado', 'value': 'EstimatedSalary'}
                    ],
                    value = 'Balance',
                    style={'fontSize': 20}
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'})
        ], className = 'row flex-display'),

        html.Div([
            html.Div([
                dcc.Graph(id='barras-chart')
            ],className='create_container2'),
            html.Div([
                dcc.Graph(id='dispersion-chart')
            ], className='create_container2')

        ], className = 'row flex-display'),

        html.Div([

            html.Div([
                dcc.Dropdown(
                    id = 'filtro-boxplot-dropdown',
                    options=[
                        {'label': 'Puntaje Crediticio', 'value': 'CreditScore'},
                        {'label': 'Balance', 'value':'Balance'},
                        {'label': 'Salario Estimado', 'value': 'EstimatedSalary'},
                        {'label': 'Edad', 'value':'Age'},
                        {'label': 'Número de Productos','value': 'NumOfProducts'},
                        {'label': 'Permanencia', 'value': 'Tenure'}
                    ],
                    value = 'CreditScore', # Valor inicial
                    style={'fontSize': 20}
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'}),

            html.Div([
                dcc.Dropdown(
                    id = 'filtro-histograma-dropdown',
                    options=[
                        {'label': 'Edad', 'value': 'Age'},
                        {'label': 'Balance', 'value': 'Balance'},
                        {'label': 'Puntaje Crediticio', 'value': 'CreditScore'}
                    ],
                    value='Age',  # Valor inicial
                    style={'fontSize': 20}
                )
            ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'center'})  
        ], className='row flex-display'),

        html.Div([
            html.Div([
                dcc.Graph(id='boxplot-chart')
            ],className='create_container2'),
            html.Div([
                dcc.Graph(id='histograma-chart')
            ], className='create_container2')

        ], className = 'row flex-display'), 

    ])

# Callback para actualizar la gráfica de barras
@app.callback(
    Output(component_id='barras-chart',component_property='figure'),
    Input(component_id='filtro-barras-dropdown',component_property='value')
)

def cargar_grafico_barras(seleccion_variable):
    # Agrupar los datos y seleciona el filtro de la variable a mostrar
    if seleccion_variable in ['NumOfProducts', 'HasCrCard', 'IsActiveMember']:
        grupo_barras_data = data.groupby([seleccion_variable, 'Exited']).size().reset_index(name='counts')

        # Gráfico de barras ('NumOfProducts', 'HasCrCard', 'IsActiveMember')
        fig_barras = px.bar(
            grupo_barras_data,
            x=seleccion_variable,
            y='counts',
            color='Exited',
            barmode='group',
            color_continuous_scale=['tomato', 'blue'],
            #title=f'Gráfico de {seleccion_variable} vs Exited',
            title='Barras en relación con el Abandono de Clientes',
            labels={'Exited': 'Abandono (0: No, 1: Sí)', 'counts': 'Cantidad de Clientes', 
                    'NumOfProducts': 'Número de Productos', 'HasCrCard': 'Tiene Tarjeta de Crédito', 
                    'IsActiveMember':'Miembro Activo'})
        fig_barras.update_layout(title_x=0.5,title_font=dict(size=20))

        return fig_barras

# Callback para actualizar la gráfica de dispersión
@app.callback(
    Output(component_id='dispersion-chart', component_property='figure'),
    Input(component_id='filtro-dispersion-dropdown', component_property='value')
)

def cargar_grafico_dispersion(seleccion_variable):
    fig_dispersion = px.scatter(
        data,
        x = 'Age',
        y = seleccion_variable,
        color = 'Exited',
        #title = f'Gráfico de {seleccion_variable} vs Edad',
        title='Dispersión en relación a la Edad',
        labels={'Age': 'Edad', seleccion_variable: seleccion_variable, 'Exited':'Abandono (0: No, 1: Sí)', 
                'Balance':'Balance', 'CreditScore': 'Puntaje Crediticio', 
                'EstimatedSalary': 'Salario Estimado'})
    fig_dispersion.update_layout(title_x=0.5, title_font=dict(size=20))

    return fig_dispersion


# Callback para actualizar la gráfica de cajas
@app.callback(
    Output(component_id='boxplot-chart', component_property='figure'),
    Input(component_id='filtro-boxplot-dropdown', component_property='value')
)
def cargar_grafico_boxplot(seleccion_variable):
    fig_boxplot = px.box(
        data,
        x = 'Exited',
        y = seleccion_variable,
        #title = f'Boxplot de {seleccion_variable} vs Exited',
        title='Diagrama de cajas en relación al Abandono de Clientes',
        labels={'Exited': 'Abandono (0: No, 1: Sí)', 'Balance':'Balance', 'Age':'Edad',
                'CreditScore': 'Puntaje Crediticio', 'EstimatedSalary': 'Salario Estimado',
                'NumOfProducts':'Número de Productos', 'Tenure':'Permanencia'},
        color='Exited',
        color_discrete_map={0: 'blue', 1: 'orange'})
    fig_boxplot.update_layout(title_x=0.5, title_font=dict(size=20))

    return fig_boxplot


# Callback para actualizar la gráfica de histograma
@app.callback(
    Output(component_id='histograma-chart', component_property='figure'),
    Input(component_id='filtro-histograma-dropdown', component_property='value')
)
def cargar_grafico_histograma(seleccion_variable):
    # Crear el gráfico de histograma
    fig_histograma = px.histogram(
        data,
        x = seleccion_variable,
        color = 'Exited',
        #nbins=30,  # Número de bins
        # title = f'Histograma de {seleccion_variable} vs Exited',
        title='Histograma en relación al Abandono de Clientes',
        #labels = {'Exited': 'Exited (0: No, 1: Sí)', seleccion_variable: seleccion_variable},
        labels={'Exited': 'Abandono (0: No, 1: Sí)', 'Balance':'Balance', 'Age':'Edad',
                'CreditScore': 'Puntaje Crediticio'},
        color_discrete_map = {0: 'blue', 1: 'purple'},
        barmode = 'overlay', # Para superponer las barras
        opacity=0.8)  # Opacidad de las barras
    
    fig_histograma.update_layout(title_x=0.5, title_font=dict(size=20))
    fig_histograma.update_traces(marker=dict(line=dict(width=1, color='black')))  # Contorno negro en las barras

    return fig_histograma



# --------Layout para Dashboard 3
def dashboard3_layout():
    return html.Div([

        html.Div([
            dcc.Dropdown(
            id='filtro-modelos-dropdown',
            options=[
                {'label': 'Random Forest', 'value': 'RF'},
                {'label': 'XGBoost', 'value': 'XGB'},
                {'label': 'Red Neuronal Perceptrón Multicapa', 'value': 'MLP'}
            ],
            value='RF',  # Valor por defecto
            style={'fontSize': 20}
            )
        ], style={'flex': 1, 'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'textAlign': 'left'}),

        # Crear contenedores para las métricas
        html.Div([
            html.H3('Resultados Métricas de Evaluación'),
            html.Div([
                html.Div([html.H3("Exactitud"), 
                        html.P(id='accuracy-metrica', style={'fontSize': '20px'})],
                        style={'border': '1px solid #ccc', 'padding': '20px', 'margin': '10px', 
                        'borderRadius': '10px', 'width': '150px', 'textAlign': 'center',
                        'box-shadow': '4px 4px 4px 4px lightgrey', 'background-color': '#F2F2F2'}),
            
                html.Div([html.H3("Precisión"), 
                        html.P(id='precision-metrica', style={'fontSize': '20px'})],
                        style={'border': '1px solid #ccc', 'padding': '20px', 'margin': '10px', 
                        'borderRadius': '10px', 'width': '150px', 'textAlign': 'center',
                        'box-shadow': '4px 4px 4px 4px lightgrey', 'background-color': '#F2F2F2'}),
            
                html.Div([html.H3("Recall"), 
                        html.P(id='recall-metrica', style={'fontSize': '20px'})],
                        style={'border': '1px solid #ccc', 'padding': '20px', 'margin': '10px', 
                        'borderRadius': '10px', 'width': '150px', 'textAlign': 'center',
                        'box-shadow': '4px 4px 4px 4px lightgrey', 'background-color': '#F2F2F2'}),
            
                html.Div([html.H3("Puntuación-F1"), 
                        html.P(id='f1-metrica', style={'fontSize': '20px'})],
                        style={'border': '1px solid #ccc', 'padding': '20px', 'margin': '10px', 
                        'borderRadius': '10px', 'width': '150px', 'textAlign': 'center',
                        'box-shadow': '4px 4px 4px 4px lightgrey', 'background-color': '#F2F2F2'}),
            
                html.Div([html.H3("AUC", ),
                        html.P(id='auc-metrica', style={'fontSize': '20px'})],
                        style={'border': '1px solid #ccc', 'padding': '20px', 'margin': '10px', 
                        'borderRadius': '10px', 'width': '150px', 'textAlign': 'center',
                        'box-shadow': '4px 4px 4px 4px lightgrey', 'background-color': '#F2F2F2'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        ], className='create_container2'),

        html.Div([
            html.Div([
                dcc.Graph(id='roc-curve'),
            ], className='create_container2'),
            html.Div([
                dcc.Graph(id='confusion-matrix'),
            ], className='create_container2')
        ], className = 'row flex-display')
    ])


# Callback para actualizar las métricas según el modelo seleccionado
@app.callback(
    #Output(component_id='metricas-evaluacion', component_property='children'),
    Output(component_id='accuracy-metrica', component_property='children'),
    Output(component_id='precision-metrica', component_property='children'),
    Output(component_id='recall-metrica', component_property='children'),
    Output(component_id='f1-metrica', component_property='children'),
    Output(component_id='auc-metrica', component_property='children'),
    Output(component_id='roc-curve', component_property='figure'),
    Output(component_id='confusion-matrix', component_property='figure'),
    Input(component_id='filtro-modelos-dropdown', component_property='value')
)

def cargar_metricas(seleccion_modelo):
    if seleccion_modelo == 'MLP':
        modelo = MLPClassifier(random_state=42, max_iter=2000)
        param_grid = {
            'hidden_layer_sizes': [(50,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.001],
            'learning_rate': ['adaptive']
        }
    elif seleccion_modelo == 'XGB':
        modelo = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100],
            'max_depth': [3],
            'learning_rate': [0.1]
        }
    else:
        modelo = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [200],
            'max_depth': [30],
            'min_samples_split': [10],
            'min_samples_leaf': [2]
        }
    
    # Realizar Grid Search
    grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_std, y_train)

    # Entrenar el mejor modelo
    best_modelo = grid_search.best_estimator_
    y_pred = best_modelo.predict(X_test_std)
    y_pred_proba = best_modelo.predict_proba(X_test_std)[:, 1]

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, y_pred_proba)

    # Crear gráfica AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Curva ROC'))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Linea', line=dict(dash='dash')))
    roc_fig.update_layout(title='Curva AUC-ROC', xaxis_title='Tasa de Falsos Positivos', yaxis_title='Tasa de Verdaderos Positivos',
                          title_x=0.5, title_font=dict(size=20))

    # Crear matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicción (No)', 'Predicción (Si)'],
        y=['Actual (No)', 'Actual (Si)'],
        colorscale='blues'
    ))
    conf_fig.update_layout(title='Matriz de Confusión', xaxis_title='Predicción', yaxis_title='Realidad',
                           title_x=0.5, title_font=dict(size=20))

    return f"{accuracy:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}", f"{auc:.2f}", roc_fig, conf_fig



# --------Layout para Dashboard 4
# Cargar el modelo y el scaler
modelo = joblib.load('venv/aplicacion/rf_model.pkl')
scaler = joblib.load('venv/aplicacion/scaler.pkl')

def dashboard4_layout():
    return html.Div([
        html.H2("Predicción de Pérdida de Clientes Bancarios", style={'textAlign': 'center'}),
        html.H3("Esta aplicación predice si un cliente tiene la tendencia de abandono de la Institución Financiera", 
                style={'textAlign': 'center'}),
        dcc.Upload(id='cargar-datos', 
                   children=html.Div([
                       'Arrastrar o ',
                       html.A('Subir CSV')
                   ]), 
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'fontSize': '22px'
                            #'backgroundColor': 'aqua'
                        },
                   ),
        html.Div([
            html.Button('Predecir', 
                    id='predecir-boton', 
                    n_clicks=0,
                    style={'font-size': '24px', 'margin': '10px', 'backgroundColor': 'navy',
                            'color': 'white'})
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center',
            'alignItems': 'center'
        }),
                    
        html.Div([
            # Contenedor 1:
            html.Div([
                html.H2('Predicción de Abandono'),
                html.P(id='prediccion-salida', style={'fontSize': '22px'})
            ], className='create_container2'),
            # Contenedor 2:
            html.Div([
                html.H2('Probabilidad de Abandono'),
                html.P(id='probabilidad-salida', style={'fontSize': '22px'})
            ], className = 'create_container2')
        ], className = 'row flex-display'),

        html.Div(id='datos-tabla', style={'margin-top': '20px'}),
        html.H3("*Nota:", style={'textAlign': 'left', 'marginLeft': '20px'}),
        html.P("\t*Geografía - España: 0, Alemania: 1, Francia: 2", style={'textAlign': 'left', 'marginLeft': '20px'}),
        html.P("\t*Género - Masculino: 0, Femenino: 1", style={'textAlign': 'left', 'marginLeft': '20px'}),
        
        # Pie de página
        html.Footer(children=[
            html.P("Este dashboard es una visualización de la predicción de abandono de clientes bancarios"),
            html.P("Desarrollado por Raham Castillo / rdcastillo47@utpl.edu.ec"),
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f9f9f9'})

    ])

def analizar_contenido(contents):
    content_type, content_string = contents.split(',')
    lectura = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(lectura.decode('utf-8')))

@app.callback(
    Output(component_id='datos-tabla', component_property='children'),
    Output(component_id='prediccion-salida', component_property='children'),
    Output(component_id='probabilidad-salida', component_property='children'),
    Input(component_id='cargar-datos', component_property='contents'),
    Input(component_id='predecir-boton', component_property='n_clicks')
)

def actualizar_salida(contents, n_clicks):
    if contents is None:
        return '', '', ''
    
    df = analizar_contenido(contents)
    #print(df)

    # Estandarizar los datos
    df_scaled = scaler.transform(df)
    
    # Predicciones
    predicciones = modelo.predict(df_scaled)
    probabilidades = modelo.predict_proba(df_scaled)  # Probabilidades de ambas clases

    if ctx.triggered_id == 'predecir-boton' and n_clicks > 0:
        pred_msg = [f'Predicción: {pred}' for pred in predicciones]
        prob_msg = [f'No Abandono (0): { prob[0]:.2f} - Abandono (1): { prob[1]:.2f}' for prob in probabilidades]

        # Renombrar columnas
        data_renombrar = df.copy()
        data_renombrar.rename(columns={
            'CreditScore': 'Puntaje Crediticio',
            'Age': 'Edad',
            'Tenure':'Permanencia',
            'NumOfProducts': 'Numero de Productos',
            'HasCrCard':'Tiene Tarjeta de Crédito',
            'IsActiveMember':'Miembro Activo',
            'EstimatedSalary': 'Salario Estimado',
            'Geography':'Geografía',
            'Gender': 'Género'
        }, inplace=True)
        #print(data_renombrar)

        # Crear tabla de datos
        data_table = DataTable(
            data=data_renombrar.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in data_renombrar.columns],
            style_header={
            'backgroundColor': 'navy',
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '18px'
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'fontSize': '18px'
            },
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            page_size=10
        )

        #predictions_div = html.Div(pred_msg, style={'margin-top': '10px'})
        #probabilities_div = html.Div(prob_msg, style={'margin-top': '10px'})

        #return html.Div(pred_msg), html.Div(prob_msg)
        #return data_table, predictions_div, probabilities_div

        #return data_table, f'{pred_msg}', f'{prob_msg}'
        return data_table, html.Div(pred_msg), html.Div(prob_msg)


    
    return '', '', ''


# Ejecutar la aplicación
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)


