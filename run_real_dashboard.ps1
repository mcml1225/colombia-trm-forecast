Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EJECUTANDO DASHBOARD CON DATOS REALES" -ForegroundColor Cyan
Write-Host "========================================
" -ForegroundColor Cyan

# Verificar datos reales
 = "data/processed/trm_historical.csv"
if (Test-Path ) {
     = Get-ChildItem 
    Write-Host "✓ Datos encontrados:" -ForegroundColor Green
    Write-Host "  Archivo: " -ForegroundColor Gray
    Write-Host "  Tamaño: 0 KB" -ForegroundColor Gray
    Write-Host "  Modificado: " -ForegroundColor Gray
    
    # Mostrar últimos valores
    Write-Host "
Últimos valores:" -ForegroundColor Yellow
    python -c "
import pandas as pd
df = pd.read_csv('data/processed/trm_historical.csv')
df['date'] = pd.to_datetime(df['date'])
print(f'  Fecha: {df[\"date\"].iloc[-1].strftime(\"%Y-%m-%d\")}')
print(f'  TRM: {df[\"trm\"].iloc[-1]:,.2f} COP/USD')
"
} else {
    Write-Host "✗ No hay datos reales. Ejecutando descarga..." -ForegroundColor Yellow
    python fetch_real_data.py
}

Write-Host "
Iniciando dashboard..." -ForegroundColor Green
Write-Host "El dashboard se abrirá en tu navegador" -ForegroundColor Yellow
Write-Host "Presiona Ctrl+C para detener
" -ForegroundColor Gray

# Ejecutar dashboard
streamlit run dashboard/app.py --server.maxUploadSize=10
