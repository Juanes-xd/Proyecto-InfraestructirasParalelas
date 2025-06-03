# Estrategia de Trading con Aprendizaje No Supervisado

## Descripción del Proyecto

Este proyecto implementa una estrategia sofisticada de trading cuantitativo utilizando técnicas de aprendizaje automático no supervisado para identificar patrones en los mercados financieros. La estrategia utiliza clustering K-Means para agrupar acciones similares basándose en indicadores técnicos y características fundamentales, luego optimiza los pesos del portafolio usando principios de la Teoría Moderna de Portafolios.

### Características Principales
- **Fuente de Datos**: Acciones del S&P 500 con 8 años de datos históricos
- **Indicadores Técnicos**: RSI, Bandas de Bollinger, ATR, MACD, Volatilidad Garman-Klass
- **Factores de Riesgo**: Integración del modelo de 5 factores Fama-French
- **Clustering**: K-Means con centroides predefinidos basados en niveles RSI
- **Optimización**: Optimización de portafolio con Frontera Eficiente
- **Benchmarking**: Comparación de rendimiento contra SPY (ETF del S&P 500)

## Conceptos Clave de Trading y Machine Learning

### Conceptos de Trading
- **RSI (Relative Strength Index)**: Oscilador que mide si una acción está sobrecomprada (>70) o sobrevendida (<30)
- **Bandas de Bollinger**: Bandas estadísticas que indican cuando el precio está alejado de su promedio móvil
- **ATR (Average True Range)**: Mide la volatilidad promedio de una acción
- **MACD**: Indicador que muestra la relación entre dos promedios móviles del precio
- **Volatilidad Garman-Klass**: Estimador de volatilidad más preciso que usa precios máximos, mínimos, apertura y cierre
- **Frontera Eficiente**: Conjunto de portafolios óptimos que ofrecen el mayor retorno esperado para cada nivel de riesgo
- **Ratio de Sharpe**: Medida de rendimiento ajustado por riesgo (retorno excedente / volatilidad)

### Conceptos de Machine Learning
- **Aprendizaje No Supervisado**: Técnica donde el algoritmo encuentra patrones en datos sin etiquetas predefinidas
- **K-Means Clustering**: Algoritmo que agrupa datos en k clusters basándose en similitudes
- **Centroides**: Puntos centrales de cada cluster que representan las características promedio del grupo
- **Factores Fama-French**: Factores académicos que explican los retornos de las acciones (mercado, tamaño, valor, rentabilidad, inversión)

## Metodología

### 1. Recolección y Preprocesamiento de Datos

La estrategia comienza descargando datos históricos de precios para todos los componentes del S&P 500:

```python
# Descargar 8 años de datos diarios para todas las acciones del S&P 500
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
df_raw = yf.download(tickers=symbols_list, start=start_date, end=end_date, auto_adjust=False)
```

**Pasos clave de preprocesamiento:**
- Los datos abarcan 8 años para asegurar suficiente contexto histórico
- `auto_adjust=False` preserva las columnas originales 'Adj Close' para cálculos precisos
- Ajustes de formato de símbolos para compatibilidad con yfinance
- Estructura DataFrame multi-nivel con índices de fecha y ticker

### 2. Cálculo de Indicadores Técnicos

La estrategia calcula múltiples indicadores técnicos para capturar diferentes dinámicas del mercado:

#### Volatilidad Garman-Klass
Un estimador de volatilidad que utiliza precios máximos, mínimos, apertura y cierre:

$$\text{Vol GK} = \frac{(\ln(\text{Máximo}) - \ln(\text{Mínimo}))^2}{2} - (2\ln(2) - 1)(\ln(\text{Cierre Adj}) - \ln(\text{Apertura}))^2$$

```python
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
```

#### RSI (Índice de Fuerza Relativa)
Oscilador de momento que mide la velocidad y cambio de los movimientos de precio:
```python
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
```

#### Bandas de Bollinger
Bandas estadísticas colocadas arriba y abajo de una media móvil:
```python
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])
```

#### Rango Verdadero Promedio (ATR)
Mide la volatilidad del mercado, normalizado para comparabilidad:
```python
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['adj close'], length=14)
    return atr.sub(atr.mean()).div(atr.std())
```

#### MACD (Convergencia y Divergencia de Medias Móviles)
Indicador de momento que sigue tendencias:
```python
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
```

### 3. Agregación Mensual y Filtrado de Liquidez

Para reducir la complejidad computacional y enfocarse en acciones líquidas:

```python
# Convertir a frecuencia mensual
data = pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                  df.unstack()[last_cols].resample('M').last().stack('ticker')], axis=1)

# Calcular promedio móvil de 5 años del volumen en dólares
data['dollar_volume'] = data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack()

# Filtrar las 150 acciones más líquidas cada mes
data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False)
data = data[data['dollar_vol_rank']<150]
```

**Beneficios:**
- Reduce el ruido de las fluctuaciones diarias
- Se enfoca en las acciones más líquidas para asegurar ejecución eficiente
- Mantiene un tamaño de universo suficiente para diversificación

### 4. Cálculo de Retornos e Ingeniería de Características

Las características de retornos multi-horizonte capturan varias dinámicas del mercado:

```python
def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]  # Retornos a 1, 2, 3, 6, 9, 12 meses
    
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
```

**Características:**
- **Recorte de valores atípicos**: Elimina valores extremos para prevenir sesgo del modelo
- **Retornos anualizados**: Comparación consistente entre diferentes horizontes temporales
- **Múltiples horizontes**: Captura momentum a corto plazo y tendencias a largo plazo

### 5. Integración de Factores Fama-French

Incorpora factores de riesgo académicos para mejor análisis ajustado por riesgo:

```python
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0]

# Calcular betas de factores móviles usando ventanas de 24 meses
betas = factor_data.groupby(level=1, group_keys=False).apply(
    lambda x: RollingOLS(endog=x['return_1m'], 
                         exog=sm.add_constant(x.drop('return_1m', axis=1)),
                         window=min(24, x.shape[0])).fit(params_only=True).params)
```

**Los cinco factores:**
- **Mkt-RF**: Prima de riesgo del mercado
- **SMB**: Pequeño menos Grande (factor de tamaño)
- **HML**: Alto menos Bajo (factor de valor)
- **RMW**: Robusto menos Débil (factor de rentabilidad)
- **CMA**: Conservador menos Agresivo (factor de inversión)

### 6. Clustering K-Means con Centroides Predefinidos

La innovación central usa centroides basados en RSI para crear clusters significativos de acciones:

```python
# Definir centroides basados en niveles RSI
target_rsi_values = [30, 45, 55, 70]  # Sobrevendido, Neutral-Bajo, Neutral-Alto, Sobrecomprado
initial_centroids = np.zeros((len(target_rsi_values), 18))  # 18 características
initial_centroids[:, 6] = target_rsi_values  # RSI es la 7ma característica (índice 6)

# Aplicar clustering K-Means
def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4, random_state=0, init=initial_centroids).fit(df).labels_
    return df
```

**Lógica de Clustering:**
- **Cluster 0** (RSI ~30): Acciones sobrevendidas, candidatos potenciales para reversión a la media
- **Cluster 1** (RSI ~45): Acciones con momentum neutral-bajo
- **Cluster 2** (RSI ~55): Acciones con momentum neutral-alto  
- **Cluster 3** (RSI ~70): Acciones sobrecompradas, candidatos para continuación de momentum

**Enfoque de la Estrategia**: Selecciona Cluster 3 (RSI ~70) basado en la hipótesis de persistencia del momentum

### 7. Optimización de Portafolio

Utiliza la Teoría Moderna de Portafolios para optimizar pesos dentro del cluster seleccionado:

```python
def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    cov = risk_models.sample_cov(prices=prices, frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),  # Máx 10% por acción
                           solver='SCS')
    
    weights = ef.max_sharpe()
    return ef.clean_weights()
```

**Características de Optimización:**
- **Objetivo**: Maximizar el ratio de Sharpe (retornos ajustados por riesgo)
- **Restricciones**: 
  - Peso mínimo: Mitad del peso igual para diversificación
  - Peso máximo: 10% por acción para prevenir concentración
- **Respaldo**: Peso igual si la optimización falla
- **Rebalanceo**: Frecuencia mensual

### 8. Evaluación de Rendimiento y Visualización

Análisis comprensivo de rendimiento contra benchmark del mercado:

```python
# Calcular retornos acumulativos
portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

# Comparar estrategia vs benchmark SPY
portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='left')
```

**Métricas de Rendimiento:**
- Cálculo de retornos diarios del portafolio
- Visualización de retornos acumulativos
- Comparación directa con SPY (ETF del S&P 500)
- Análisis de rendimiento ajustado por riesgo

## Detalles de Implementación

### Librerías Requeridas
```python
# Datos y Análisis
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

# Análisis Técnico
import pandas_ta

# Machine Learning
from sklearn.cluster import KMeans

# Optimización de Portafolio  
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Análisis Estadístico
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# Visualización
import matplotlib.pyplot as plt
```

### Orden de Ejecución (Importante)
1. **Definir centroides primero** (Celda 28): Debe ejecutarse antes del clustering K-Means
2. **Ejecutar clustering K-Means** (Celda 24): Usa centroides predefinidos
3. **Funciones de visualización** (Celdas 25-26): Graficar distribuciones de clusters
4. **Bucle de optimización de portafolio**: Rebalanceo mensual y seguimiento de rendimiento

### Estructuras de Datos Clave

**DataFrame de Características (`data`)**:
- Multi-índice: (fecha, ticker)
- 18 características incluyendo indicadores técnicos, retornos y betas de factores
- Frecuencia mensual de agregación diaria
- Las 150 acciones más líquidas por mes

**DataFrame de Portafolio (`portfolio_df`)**:
- Frecuencia diaria para cálculo preciso de retornos
- Comparación de retornos de estrategia vs benchmark
- Seguimiento de rendimiento acumulativo

## Rendimiento de la Estrategia

La estrategia demuestra:
- **Enfoque sistemático**: Selección y optimización basada en reglas
- **Gestión de riesgo**: Restricciones de diversificación y exposición a factores
- **Adaptabilidad**: Rebalanceo mensual con datos frescos del mercado
- **Transparencia**: Visualización clara y benchmarking

## Notas Técnicas

### Consideraciones de Calidad de Datos
- **Parámetro auto-adjust**: Establecido en `False` para preservar columnas 'Adj Close'
- **Manejo de datos faltantes**: Forward-fill para betas de factores, dropna para clustering
- **Tratamiento de valores atípicos**: Winsorización de retornos en percentiles 0.5% y 99.5%

### Robustez de Optimización
- **Selección de solver**: Solver SCS para estabilidad numérica
- **Mecanismos de respaldo**: Peso igual cuando la optimización falla
- **Validación de restricciones**: Observaciones mínimas requeridas para regresión de factores

### Consideraciones de Rendimiento
- **Eficiencia computacional**: Frecuencia mensual reduce tiempo de procesamiento
- **Gestión de memoria**: Procesamiento en chunks para datasets grandes
- **Manejo de errores**: Degradación elegante con logging informativo

## Explicaciones Adicionales de Conceptos

### Conceptos Avanzados de Trading

**Momentum vs Reversión a la Media**:
- **Momentum**: Tendencia de que los precios que suben sigan subiendo (base para seleccionar Cluster 3)
- **Reversión a la Media**: Tendencia de que los precios extremos vuelvan a su promedio histórico

**Volatilidad Garman-Klass**:
- Más precisa que la volatilidad tradicional porque usa información intradiaria
- Considera la diferencia entre máximos/mínimos y apertura/cierre
- Útil para medir el "ruido" o incertidumbre en el precio

**Factores de Riesgo Fama-French**:
- Explican ~95% de la variación en retornos de acciones
- Permiten evaluar si los retornos se deben a skill o exposición a riesgos conocidos
- Base para modelos de pricing de activos modernos

### Conceptos Avanzados de Machine Learning

**Clustering No Supervisado**:
- No requiere etiquetas predefinidas (vs clasificación supervisada)
- Descubre patrones ocultos en los datos automáticamente
- K-Means minimiza la varianza dentro de cada cluster

**Centroides Predefinidos**:
- Innovación que combina conocimiento de dominio (trading) con ML
- Asegura que los clusters tengan significado financiero interpretable
- Mejor que inicialización aleatoria para aplicaciones financieras

**Ventana Móvil (Rolling Window)**:
- Técnica para calcular estadísticas usando períodos fijos que se "mueven" en el tiempo
- Captura dinámicas cambiantes del mercado
- Evita look-ahead bias (usar información futura)

Esta implementación proporciona un marco integral para desarrollar y probar estrategias de trading cuantitativo usando técnicas de aprendizaje no supervisado, con gestión de riesgo adecuada y metodologías de evaluación de rendimiento.

