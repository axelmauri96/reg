import streamlit as st
import pandas as pd
import joblib

st.title("Mantenimiento Predictivo - Camanchaca")

uploaded_file = st.file_uploader("Sube un archivo CSV con datos de sensores", type=["csv"])

if uploaded_file:
    df_app = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df_app.head())

    modelo = joblib.load("modelo_rf.joblib")

    if st.button("Predecir fallas"):
        # Ensure the columns match the training data features
        features_safe = ["motor_temp","bearing_temp","vibration","load","ambient_temp",
                         "operating_hours","delta_temp","relative_load","vib_load_index"]
        
        # Add missing columns with default values (e.g., 0 or median/mean from training data)
        # Or, ideally, preprocess the uploaded data using the same pipeline as the training data
        
        # For now, let's assume the uploaded data has the same columns and order as the training data
        # A more robust solution would involve applying the same preprocessing steps used during training

        # Check if required columns are in the uploaded file
        missing_cols = [col for col in features_safe if col not in df_app.columns]
        if missing_cols:
            st.error(f"Missing required columns in the uploaded file: {', '.join(missing_cols)}")
        else:
            # Select only the safe features for prediction
            df_app_processed = df_app[features_safe]

            # Apply the same numerical preprocessing (imputation and scaling)
            # Ideally, save and load the preprocessor from training
            # For this example, we'll re-apply a basic imputer and scaler (less robust)
            # A better approach: save the entire pipeline including the preprocessor
            
            # Load the numerical preprocessor used during training
            # Assuming the pipeline saved in modelo_rf.joblib includes the preprocessor
            # If not, you would need to save the preprocessor separately

            # Let's assume the loaded 'modelo' is the full pipeline
            # If 'modelo' is just the RandomForestClassifier, you would need the preprocessor separately
            
            # If the loaded 'modelo' is the pipeline including the preprocessor:
            try:
                # Assuming the pipeline step for preprocessing is named 'prep'
                # You might need to inspect the pipeline steps to confirm the name
                preprocessor = modelo.named_steps['prep']
                df_app_transformed = preprocessor.transform(df_app_processed)
                
                # Predict using the model step in the pipeline
                predicciones = modelo.named_steps['model'].predict(df_app_transformed)
                st.write("Resultados de la predicción:")
                st.write(predicciones)

            except KeyError:
                st.error("Could not find the 'prep' step in the loaded model pipeline. Ensure the saved model includes the preprocessor or load the preprocessor separately.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


# For executing in Colab:
# !streamlit run app.py
