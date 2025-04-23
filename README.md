<div align="center">
  
# **Bienvenidos al Software SIFIA**

</div>

## **Descripción de la aplicación**

El software SIFIA es una aplicación web tipo dashboard para la predicción de pérdida de clientes bancarios. Su principal función es analizar el comportamiento de los clientes, de tal manera, que los modelos de aprendizaje automático presenten una predicción confiable ante el posible abandono de futuros clientes.

## **Instalación de la aplicación**

Para que la aplicación funcione correctamente en un dispositivo de escritorio, es necesario que cumpla con los siguientes requisitos:
- Sistema Operativo Windows 7 o superior
- Visual Studio Code (1.99.3)
- Versión de Python (3.10)
- Navegador web Google Chrome (preferencia)

Para la puesta en marcha del software SIFIA, se recomienda crear un entorno virtual, de tal manera que no exista conflicto con versiones de otros proyectos para el desarrollo correcto de esta aplicación.

## **Creación del entorno virtual y despliegue de la aplicación SIFIA**
Se abre Visual Studio Code en la carpeta donde se creará el entorno virtual. Se ejecuta el comando que se detalla a continuación desde el bash de vscode. Esto creará un entorno virtual "venv" con todos los archivos del entorno virtual.
- python -m venv venv

Una vez creado el entorno virtual, es necesario activarlo con:
- source venv/Scripts/activate

Se genera una carpeta con el nombre de “aplicacion" que servirá para guardar el proyecto. Luego se clona el directorio de github.
- git clone https://github.com/rahamdair/AplicacionSIFIA.git

Posteriormente, instalamos todas las librerías que permiten ejecutar todas las funciones de la aplicación SIFIA desde el bash asegurándose de que se encuentre en la raíz de la aplicación (carpeta del sistema) que conceda accesos a la ruta de la instalación del archivo “requirements.txt”. Ejecutamos la siguiente instrucción.
- pip install -r requirements.txt

Para desplegar la página web tipo dashboard en el servidor podemos ejecutar desde la terminal o usar vscode asegurándose que esté en la raíz de la aplicación “app_dashborad.py”.
Una vez inicializado el despliegue, se mostrará la siguiente dirección del servidor http://127.0.0.1:8050/, misma que conduce a la página de la aplicación.

Dentro de la carpeta se encontrará los diferentes modelos de aprendizaje automático utilizados para experimentar con algoritmos de ML y DL. Dentro de estos se seleccionaron los que mejor desempeño presentaron para desarrollar esta solución.

## **Uso de la aplicación**

Al hacer clic en la dirección url, se presenta la vista 1: **Datos de clientes** del software SIFIA. En esta pestaña se puede utilizar los filtros de selección: género y geografía para visualizar el histórico del conjunto de datos. Así como acciones a validar: Análisis, Evaluación de modelos y Modelo de predicción.  La vista 2: **Análisis**, que se muestra en la figura 2 posee diferentes filtros de selección que permiten relacionar el comportamiento de las variables que inciden en el abandono de los clientes. 

La vista 3: **Evaluación de modelos**, posee un filtro de selección para presentar los resultados de tres modelos de IA: Random Forest (RF), Extreme Gradient Boosting (XGBoost) y una Red Neuronal Perceptrón Multicapa (MLPNN) los cuales muestran los resultados de las métricas de evaluación: Exactitud (Accuracy), Precisión (Precision), Recall, Puntuación-F1 (F1 Score) y el AUC. Estas métricas contribuyen a contrarrestar el desempeño de los algoritmos de aprendizaje automático. La vista 4: **Modelos de predicción**, muestran el resultado de la predicción de un nuevo cliente a través de un archivo en formato csv “Dataset_prueba1.csv”.






