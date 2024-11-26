from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Cargar datos históricos (ejemplo con un CSV)
data = pd.read_csv('datos_delictivos.csv')

# Preprocesamiento de datos
data['hora'] = pd.to_datetime(data['hora'], format='%H:%M').dt.hour
X = data[['tipo_delito', 'ubicacion', 'hora']]  # Variables independientes
y = data['riesgo']  # Variable dependiente (etiqueta)

# Codificación de datos categóricos
X = pd.get_dummies(X)

# Dividir datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

@app.route('/predecir', methods=['POST'])
def predecir():
    datos_usuario = request.json
    entrada = pd.DataFrame([datos_usuario])
    entrada = pd.get_dummies(entrada).reindex(columns=X.columns, fill_value=0)
    prediccion = modelo.predict(entrada)
    return jsonify({'riesgo': prediccion[0]})

if __name__ == '__main__':
    app.run(debug=True)
