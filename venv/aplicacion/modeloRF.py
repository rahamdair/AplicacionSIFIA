import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

# Carga del dataset
data = pd.read_csv('venv/aplicacion/datasets/Churn_Modelling_escenario1.csv')
#data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
X = data.drop('Exited', axis=1)
y = data['Exited']

# Convertir variables categóricas a numéricas
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# GridSearch para Random Forest
param_grid = {
    'n_estimators': [200],
    'max_depth': [30],
    'min_samples_split': [10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled, y)

# Guardar el modelo y el scaler
joblib.dump(grid_search.best_estimator_, 'venv/aplicacion/rf_model.pkl')
joblib.dump(scaler, 'venv/aplicacion/scaler.pkl')
