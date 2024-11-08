# importar las librerías a utilizar
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# crear la aplicación
app = Flask(__name__)
CORS(app)  # Habilitar CORS para toda la aplicación

# cargar el modelo ya entrenado
modelo = joblib.load('modelo_random_forest.pkl')

# código a ejecutar cuando se mande el formulario
@app.route('/clasificar', methods=['POST'])
def clasificar():
    # Obtener datos enviados por el frontend en formato JSON
    datos = request.get_json()

    # Crear un DataFrame con los datos recibidos
    nuevo_registro = pd.DataFrame([datos])

    # mostrar en consola el registro creado

    # Realizar la predicción
    prediccion = modelo.predict(nuevo_registro)

    # Devolver la predicción como una respuesta JSON
    print("prediccon: ")
    print(prediccion[0])
    return jsonify({'prediccion': int(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True)