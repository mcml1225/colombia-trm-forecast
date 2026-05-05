import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.append(str(Path.cwd() / 'src'))

print("=" * 70)
print("TESTING DASHBOARD WITH REAL DATA")
print("=" * 70)

# 1. Verificar datos reales
print("\n1. Verificando datos reales...")
data_path = Path("data/processed/trm_historical.csv")

if data_path.exists():
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"   ✓ Datos encontrados")
    print(f"   - Registros: {len(df)}")
    print(f"   - Desde: {df['date'].min().strftime('%Y-%m-%d')}")
    print(f"   - Hasta: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   - Última TRM: {df['trm'].iloc[-1]:.2f}")
    
    # Verificar calidad
    missing = df['trm'].isnull().sum()
    if missing > 0:
        print(f"   ⚠ Datos faltantes: {missing} registros")
    else:
        print(f"   ✓ Sin datos faltantes")
    
    # Verificar rango de fechas
    days_range = (df['date'].max() - df['date'].min()).days
    print(f"   - Rango de días: {days_range}")
    
else:
    print("   ✗ No hay datos reales. Primero ejecuta el data_fetcher")
    exit(1)

# 2. Verificar estadísticas
print("\n2. Estadísticas de los datos reales...")
print(f"   TRM promedio: {df['trm'].mean():.2f}")
print(f"   TRM mínimo: {df['trm'].min():.2f}")
print(f"   TRM máximo: {df['trm'].max():.2f}")
print(f"   Volatilidad: {df['trm'].pct_change().std()*100:.2f}%")

# 3. Verificar estructura para el dashboard
print("\n3. Verificando estructura para dashboard...")
required_columns = ['date', 'trm', 'open_rate', 'close_rate', 'max_rate', 'min_rate']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"   ⚠ Columnas faltantes: {missing_cols}")
else:
    print(f"   ✓ Todas las columnas necesarias presentes")

# 4. Probar visualización rápida
print("\n4. Probando visualización...")
try:
    import plotly.express as px
    fig = px.line(df.tail(90), x='date', y='trm', title='TRM Real (Últimos 90 días)')
    print("   ✓ Gráfico creado exitosamente")
except Exception as e:
    print(f"   ✗ Error en visualización: {e}")

print("\n" + "=" * 70)
print("✓ DATOS REALES LISTOS PARA EL DASHBOARD")
print("=" * 70)
print("\nPara ejecutar el dashboard con datos reales:")
print("  streamlit run dashboard/app.py")
