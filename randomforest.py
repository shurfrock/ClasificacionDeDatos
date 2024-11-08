# importar las librerías a utilizar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# leer el dataset
dataset = pd.read_csv('/Users/Abraham/Documents/9no_Semestre/Clasificacion_inteligente_de_datos/DataSet/archive/Student_performance_data.csv')

# eliminar las columnas innecesarias
dataset = dataset.drop(columns=['StudentID','Gender','Ethnicity'])

# separar las variables independientes (X) de la variable dependiente (Y)
y = dataset['GradeClass']
X = dataset.drop(columns=['GradeClass'])

# barajar el dataset, obtener el grupo de entrenamiento (80%) y el grupo de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# obtener el modelo de random forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# entrenar el modelo con el grupo de entrenamiento
modelo.fit(X_train, y_train)

# guardar el modelo entrenado
joblib.dump(modelo, 'modelo_random_forest.pkl')

print("Modelo entrenado y guardado exitosamente.")