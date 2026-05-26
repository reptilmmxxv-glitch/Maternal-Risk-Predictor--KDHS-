# =============================================================================
# guardar_modelo.py
# Ejecutar este script al FINAL del notebook de entrenamiento (Colab)
# para serializar el modelo, el scaler y las columnas.
# =============================================================================

import joblib
import os

# Crear carpeta models si no existe
os.makedirs('models', exist_ok=True)

# ── Guardar el modelo principal (Gradient Boosting) ──────────────────────────
# Asegurarse de que gb_model, scaler y X esten cargados en memoria
joblib.dump(gb_model,       'models/modelo.pkl')
joblib.dump(scaler,         'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/columnas.pkl')

print("Modelo serializado correctamente en la carpeta /models")
print("Columnas guardadas:", len(X.columns.tolist()))
print("Archivos generados:")
print("  models/modelo.pkl")
print("  models/scaler.pkl")
print("  models/columnas.pkl")
