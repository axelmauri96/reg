
# Cómo usar (local, Windows/Mac/Linux)

1) Crea una carpeta de trabajo y descomprime/guarda estos 3 archivos dentro:
   - train_no_leak.py
   - app.py
   - requirements.txt

2) (Recomendado) Crea un entorno virtual e instala dependencias:
   python -m venv .venv
   # Activar:
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt

3) Entrena y guarda modelos (ajusta la ruta al CSV real):
   python train_no_leak.py --csv dataset_tornillos_camanchaca_listo_modelo.csv

   Salidas:
     models/rf_cls_pipeline.joblib
     models/rf_reg_pipeline.joblib
     outputs/metrics_summary_no_leak.json

4) Ejecuta la app:
   streamlit run app.py

5) En la app, sube el CSV (mismo esquema que entrenamiento).

Notas importantes:
- Evité usar `!streamlit run app.py` porque ese es sintaxis de Jupyter/Colab. En local se usa `streamlit run app.py` en la terminal.
- El código guarda y carga *pipelines completos* (preprocesamiento + modelo), evitando errores de columnas y "joblib not found".
