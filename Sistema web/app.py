"""
Flask — Predicción de Riesgo de Mortalidad Materna
ACIF104 — Aprendizaje de Máquinas
"""

from flask import Flask, request, jsonify, render_template, Response, session, redirect, url_for
from functools import wraps
import base64
import joblib
import pandas as pd
import numpy as np
import csv
import os
import sys
import traceback
from datetime import datetime

# ── PARCHE DE CONTENCIÓN PARA SHAP ──────────────────────────────────────────
try:
    import shap
except ImportError:
    print("Aviso: SHAP no es compatible con NumPy 2.x en este entorno, pero la app continuará ejecutándose.", flush=True)
    shap = None
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'mmr-acif104-key-2024'

# ── CARGAR MODELO Y TRANSFORMADORES CON CAPTURA DE ERRORES FORZADA ───────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'modelo.pkl')
COLS_PATH  = os.path.join(os.path.dirname(__file__), 'models', 'columnas.pkl')
LOG_PATH   = os.path.join(os.path.dirname(__file__), 'logs', 'predicciones.csv')

try:
    print("Iniciando la carga de serializaciones (.pkl)...", flush=True)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo del modelo en: {MODEL_PATH}")
    if not os.path.exists(COLS_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo de columnas en: {COLS_PATH}")

    model    = joblib.load(MODEL_PATH)
    print("-> Archivo 'modelo.pkl' cargado exitosamente.", flush=True)
    
    columnas = joblib.load(COLS_PATH)
    print("-> Archivo 'columnas.pkl' cargado exitosamente.", flush=True)

except Exception as e:
    print(f"\n--- ERROR CRÍTICO DURANTE LA INICIALIZACIÓN ---", flush=True)
    print(f"Tipo de excepción: {type(e).__name__}", flush=True)
    print(f"Mensaje de error: {str(e)}", flush=True)
    print("Detalle del Traceback:", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

# ── UMBRAL ÓPTIMO ─────────────────────────────────────────────────────────────
UMBRAL = 0.28

# ── VARIABLES NOMINALES ───────────────────────────────────────────────────────
NOMINAL_VARS = ['marital_status', 'urbanization', 'county', 'place_delivery', 'skilled_attendant']

# ── LABELS EN ESPAÑOL ─────────────────────────────────────────────────────────
SPANISH_LABELS = {
    'age':                     'Edad (años)',
    'parity':                  'Número de partos previos',
    'education':               'Nivel educativo',
    'anc_visits':              'Visitas prenatales (ANC)',
    'c_section':               'Cesárea',
    'postnatal48h':            'Atención postnatal 48h',
    'age_first_birth':         'Edad al primer parto',
    'birth_interval':          'Intervalo entre partos (meses)',
    'wealth_quintile':         'Quintil de riqueza',
    'household_size':          'Tamaño del hogar',
    'distance_facility':       'Distancia al centro de salud (km)',
    'health_facility_density': 'Densidad de establecimientos de salud',
    'poverty_rate':            'Tasa de pobreza del condado (%)',
    'sba_coverage':            'Cobertura de personal calificado (%)',
    'county_edu_index':        'Índice educacional del condado',
    'hiv_prev':                'Prevalencia VIH en la región (%)',
}

# ── SESSION AUTH DECORATOR ───────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return render_template('login.html')
        return f(*args, **kwargs)
    return decorated

# ── PREPROCESAMIENTO ──────────────────────────────────────────────────────────
def preprocesar(datos: dict) -> pd.DataFrame:
    edu_map = {'Primary': 1, 'Secondary': 2, 'Higher': 3}
    datos['education'] = edu_map.get(datos.get('education', 'Primary'), 1)
    df = pd.DataFrame([datos])
    nominal_presentes = [v for v in NOMINAL_VARS if v in df.columns]
    if nominal_presentes:
        df = pd.get_dummies(df, columns=nominal_presentes, drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    for col in columnas:
        if col not in df.columns:
            df[col] = 0
    df = df[columnas].astype(float)
    return df

def calcular_shap(df: pd.DataFrame) -> list:
    if shap is None:
        return []
        
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        nombres = df.columns.tolist()
        pares = sorted(zip(nombres, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        return [
            {
                'variable': SPANISH_LABELS.get(nombre, nombre),
                'efecto': round(float(valor), 4),
                'direccion': 'aumenta el riesgo' if valor > 0 else 'reduce el riesgo'
            }
            for nombre, valor in pares
        ]
    except Exception as e:
        print(f'Error SHAP: {e}', flush=True)
        return []

def registrar_log(datos: dict, probabilidad: float, clasificacion: int):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        fila = {'timestamp': datetime.now().isoformat(), 'probabilidad': probabilidad, 'clasificacion': clasificacion, **datos}
        archivo_existe = os.path.exists(LOG_PATH)
        with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fila.keys())
            if not archivo_existe:
                writer.writeheader()
            writer.writerow(fila)
    except Exception as e:
        print(f'Error al registrar log local: {e}', flush=True)

# ── RUTAS ─────────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        datos = request.get_json()
        if not datos:
            return jsonify({'error': 'No se recibieron datos'}), 400
        df = preprocesar(datos)
        probabilidad = float(model.predict_proba(df)[0][1])
        clasificacion = int(probabilidad >= UMBRAL)
        nivel_riesgo = 'Alto' if probabilidad >= 0.5 else 'Moderado' if probabilidad >= UMBRAL else 'Bajo'
        factores_shap = calcular_shap(df)
        registrar_log(datos, probabilidad, clasificacion)
        return jsonify({
            'probabilidad': round(probabilidad * 100, 1),
            'clasificacion': clasificacion,
            'nivel_riesgo': nivel_riesgo,
            'umbral_usado': UMBRAL,
            'factores_shap': factores_shap
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/columns', methods=['GET'])
@login_required
def get_columns():
    field_types = {}
    for col in columnas:
        if col in NOMINAL_VARS:
            field_types[col] = 'categorical'
        elif col in ['c_section', 'postnatal48h', 'skilled_attendant']:
            field_types[col] = 'binary'
        else:
            field_types[col] = 'numeric'
    return jsonify({
        'columns': columnas.tolist(),
        'numeric_cols': [],
        'labels': SPANISH_LABELS,
        'field_types': field_types
    })

@app.route('/health')
def health():
    return jsonify({'estado': 'activo', 'umbral': UMBRAL, 'columnas': len(columnas)})

# ── LOGIN ROUTES ──────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('index'))
        return 'Credenciales inválidas', 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)