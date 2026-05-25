# Maternal-Risk-Predictor--KDHS-
![Vista del sistema](preview_.png)

 
# Sistema de Predicción de Riesgo de Mortalidad Materna

Desarrollado por **C. Rozas Balboa**  
Alumno de Ingeniería Civil Informática  
Facultad de Ingeniería, Universidad Andrés Bello  
Curso: Aprendizaje de Máquinas — ACIF104 | Mayo 2026  

---

## Descripción del Proyecto

Este proyecto consiste en el desarrollo de un sistema predictivo orientado a la  
identificación del riesgo de mortalidad materna a partir del conjunto de datos  
KDHS 2022 (Kenya Demographic and Health Survey). Mediante la aplicación de  
técnicas de Aprendizaje Automático, se analizan 22 atributos de naturaleza  
sociodemográfica y clínica para ofrecer una herramienta funcional de apoyo en  
la toma de decisiones médicas.

El sistema integra un pipeline completo de machine learning con un backend en  
Flask y un frontend web en Bootstrap, permitiendo la predicción en tiempo real  
con explicabilidad incorporada mediante SHAP.

> **Aviso de alcance:** Este modelo fue entrenado exclusivamente con datos de  
> Kenya. Sus predicciones no son válidas ni generalizables a otros contextos  
> geográficos o sanitarios, incluyendo Chile u otros países de América Latina.

---

## Resultados del Análisis

De acuerdo con el análisis realizado, se compararon tres modelos de Machine  
Learning (Regresión Logística, Random Forest y Gradient Boosting) y tres  
arquitecturas de Deep Learning (MLP básico, MLP con Dropout y BatchNormalization,  
y MLP profundo con regularización L2).

El modelo de Gradient Boosting fue seleccionado como modelo principal del  
sistema por presentar el mejor F1-score tras la optimización del umbral de  
decisión (0,34), mientras que la Regresión Logística obtuvo el mayor AUC-ROC  
individual (0,634). El umbral de decisión fue optimizado mediante la curva  
Precision-Recall, pasando de 0,5 por defecto a 0,28, lo que elevó el recall  
de la clase positiva de 0,07 a 0,46, priorizando la detección de casos de  
riesgo real sobre la precisión general.

Los resultados indican que la tasa de pobreza regional, la paridad, la edad  
materna, el número de controles prenatales y la distancia al centro de salud  
actúan como los predictores de mayor relevancia. Para asegurar la transparencia  
de las predicciones, se incorporó la metodología SHAP con TreeExplainer,  
permitiendo interpretar la contribución de cada variable en los resultados  
individuales.

| Modelo              | AUC-ROC | Recall (cl.1) | F1 (cl.1) | Umbral óptimo |
|---------------------|---------|---------------|-----------|---------------|
| Regresión Logística | 0.634   | 0.45          | 0.33      | 0.28          |
| Random Forest       | 0.611   | 0.48          | 0.32      | 0.20          |
| Gradient Boosting   | 0.620   | 0.46          | 0.34      | 0.25          |
| DL2 Dropout+BN      | 0.622   | 0.47          | 0.33      | 0.22          |

El AUC moderado es consistente con la baja correlación individual de las  
variables con el objetivo (máxima r = 0,106), lo que indica que la limitación  
es la capacidad informativa de los datos disponibles y no la arquitectura del  
modelo.

---

## Interfaz del Sistema

La operacionalización del modelo se implementó mediante una interfaz web  
diseñada para maximizar la agilidad en entornos clínicos. El sistema incluye:

- **Pantalla de login** con autenticación por usuario y contraseña, que controla  
  el acceso al sistema y protege todas las rutas de predicción.  
- **Formulario de entrada** organizado en cuatro secciones clínicas: datos  
  demográficos, historia obstétrica, datos del hogar e indicadores regionales.  
- **Autocompletado por condado:** al seleccionar el condado, los cinco  
  indicadores regionales (prevalencia VIH, tasa de pobreza, cobertura de  
  personal calificado, densidad de establecimientos e índice educacional) se  
  completan automáticamente con los valores reales del KDHS 2022.  
- **Indicador visual de riesgo** categorizado en niveles Bajo, Moderado y Alto,  
  con código de color y probabilidad en porcentaje.  
- **Factores SHAP** de la predicción individual, representados con barras  
  horizontales de colores que indican la dirección e intensidad de cada variable.  
- **Aviso de alcance geográfico** visible antes de cada predicción, advirtiendo  
  que el modelo es válido solo para la población de referencia de Kenya.  

El sistema prioriza la privacidad del paciente al procesar la información de  
manera efímera, sin almacenamiento persistente de datos sensibles. Cada  
predicción se registra únicamente en un log local con timestamp, probabilidad  
y clasificación, para fines de monitoreo del modelo en producción.

El prototipo de la interfaz fue diseñado previamente en Figma:  
https://www.figma.com/design/aVGIlXW6VzxPD0Xdb3HGcS/Sistema-de-Prediccion-de-Riesgo-de-Mortalidad-Materna  

---

## Arquitectura Técnica

La infraestructura de la aplicación se compone de tres capas:

**Frontend:** HTML5 y Bootstrap 5 garantizan la responsividad en diferentes  
dispositivos. JavaScript vanilla gestiona la comunicación asíncrona con el  
backend vía fetch() y renderiza los resultados dinámicamente sin recargar  
la página.

**Backend:** Microframework Flask de Python gestiona la carga del modelo  
entrenado, aplica el preprocesamiento idéntico al entrenamiento (mapeo ordinal,  
One-Hot Encoding con drop_first=True y alineación de columnas), ejecuta la  
inferencia con el umbral optimizado y calcula los valores SHAP individuales.  
Expone tres endpoints: /login (autenticación), /predict (inferencia) y /health  
(monitoreo).

**Motor predictivo:** Modelo Gradient Boosting serializado con joblib, entrenado  
sobre datos balanceados con SMOTE. Las columnas del modelo se guardan en el  
mismo orden del entrenamiento para garantizar la reproducibilidad exacta de  
los resultados.

---

## Estructura del Repositorio

```
Maternal-Risk-Predictor--KDHS-/
|
|-- notebooks/
|   |-- Mortalidad_materna_S9.ipynb     # Pipeline completo de ML
|
|-- backend/
|   |-- app.py                          # Servidor Flask
|   |-- guardar_modelo.py               # Script de serialización del modelo
|   |-- requirements.txt                # Dependencias Python
|   |-- models/                         # Archivos serializados (generados por el usuario)
|   |   |-- modelo.pkl
|   |   |-- columnas.pkl
|   |-- logs/                           # Registro de predicciones (generado automáticamente)
|   |   |-- predicciones.csv
|   |-- templates/
|       |-- login.html                  # Pantalla de autenticación
|       |-- index.html                  # Formulario de predicción
|
|-- data/
|   |-- kdhs_2022_data_maternal_mortality.csv
|
|-- README.md
```

---

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/reptilmmxxv-glitch/Maternal-Risk-Predictor--KDHS-.git
cd Maternal-Risk-Predictor--KDHS-
```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac / Linux:
```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r backend/requirements.txt
```

### 4. Generar los archivos del modelo desde Colab

Ejecuta el notebook `notebooks/Mortalidad_materna_S9.ipynb` completo en  
Google Colab (`Ctrl + F9`). Al finalizar, ejecuta esta celda:

```python
import joblib, os
os.makedirs('models', exist_ok=True)
joblib.dump(gb_model,            'models/modelo.pkl',   protocol=2)
joblib.dump(X.columns.tolist(),  'models/columnas.pkl', protocol=2)
print("Modelo guardado correctamente")
```

Descarga los archivos desde el panel de archivos de Colab y colócalos en  
`backend/models/`.

### 5. Ejecutar el servidor

```bash
cd backend
python app.py
```

Abre `http://localhost:5000` en el navegador.

### 6. Credenciales de acceso

```
Usuario:    admin
Contraseña: admin
```

---

## Tecnologías Utilizadas

| Componente    | Tecnología         | Versión |
|---------------|--------------------|---------|
| Lenguaje      | Python             | 3.12    |
| ML            | scikit-learn       | 1.6.1   |
| Deep Learning | TensorFlow / Keras | 2.x     |
| Explicabilidad| SHAP               | 0.45    |
| Serialización | joblib             | 1.4     |
| Backend       | Flask              | 3.0     |
| Frontend      | Bootstrap          | 5.3     |
| Datos         | pandas, numpy      | —       |
