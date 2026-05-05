import subprocess
import sys
from pathlib import Path

print("=" * 50)
print("EJECUTANDO DASHBOARD CON DATOS REALES")
print("=" * 50)

# Verificar datos
data_path = Path("data/processed/trm_historical.csv")
if not data_path.exists():
    print("No hay datos reales. Creando datos de ejemplo...")
    from src.utils import create_sample_data
    df = create_sample_data(200)
    df.to_csv(data_path, index=False)
    print(f"Datos creados: {len(df)} registros")
else:
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"Datos encontrados: {len(df)} registros")
    print(f"Última TRM: {df['trm'].iloc[-1]:,.2f} COP/USD")

print("\nIniciando Streamlit dashboard...")
print("El dashboard se abrirá en tu navegador")
print("Presiona Ctrl+C para detener\n")

# Ejecutar streamlit
subprocess.run(["streamlit", "run", "dashboard/app.py"])
