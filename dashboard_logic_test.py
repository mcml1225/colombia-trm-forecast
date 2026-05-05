import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path.cwd() / 'src'))

print("=" * 70)
print("PROBANDO LÓGICA DEL DASHBOARD CON DATOS REALES")
print("=" * 70)

# Cargar datos reales
data_path = Path("data/processed/trm_historical.csv")
if not data_path.exists():
    print("✗ No hay datos. Ejecuta fetch_real_data.py primero")
    exit(1)

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

print(f"\n✓ Datos cargados: {len(df)} registros")

# 1. Probar métricas del dashboard
print("\n[1] Métricas del dashboard:")
current_trm = df['trm'].iloc[-1]
daily_change = df['trm'].pct_change().iloc[-1] * 100
max_30d = df['trm'].tail(30).max()
min_30d = df['trm'].tail(30).min()
volatility = df['trm'].tail(30).pct_change().std() * 100

print(f"   TRM Actual: {current_trm:,.2f}")
print(f"   Cambio Diario: {daily_change:+.2f}%")
print(f"   Máximo 30 días: {max_30d:,.2f}")
print(f"   Mínimo 30 días: {min_30d:,.2f}")
print(f"   Volatilidad 30d: {volatility:.2f}%")

# 2. Probar tendencias
print("\n[2] Análisis de tendencias:")
# Tendencia de 7 días
ma_7 = df['trm'].tail(7).mean()
ma_30 = df['trm'].tail(30).mean()
trend = "ALZA" if ma_7 > ma_30 else "BAJA"
print(f"   Tendencia (7d vs 30d): {trend}")
print(f"   Promedio 7 días: {ma_7:.2f}")
print(f"   Promedio 30 días: {ma_30:.2f}")

# 3. Probar estacionalidad
print("\n[3] Análisis por día de semana:")
df['day_name'] = df['date'].dt.day_name()
daily_avg = df.groupby('day_name')['trm'].mean().sort_values()
for day, value in daily_avg.items():
    print(f"   {day}: {value:,.2f}")

# 4. Probar detección de anomalías
print("\n[4] Detección de anomalías:")
mean = df['trm'].mean()
std = df['trm'].std()
anomalies = df[abs(df['trm'] - mean) > 3 * std]
print(f"   Anomalías detectadas: {len(anomalies)}")
if len(anomalies) > 0:
    print(f"   Valores anómalos: {anomalies['trm'].tolist()}")

# 5. Probar predicción simple
print("\n[5] Predicción simple:")
from sklearn.linear_model import LinearRegression
X = np.arange(len(df)).reshape(-1, 1)
y = df['trm'].values
model = LinearRegression()
model.fit(X[-90:], y[-90:])
next_day = model.predict([[len(df)]])[0]
print(f"   Predicción siguiente día: {next_day:.2f}")
print(f"   Cambio esperado: {((next_day - current_trm)/current_trm*100):+.2f}%")

print("\n" + "=" * 70)
print("✓ DASHBOARD LOGIC TEST COMPLETED")
print("=" * 70)
print("\nEl dashboard debería funcionar correctamente con estos datos")
