# Tarea 5 
## Cómputo Evolutivo 2026-1

**Equipo:** TheKingdomPinguin
**Integrantes:** 

 - Castillo Hernández Antonio - 320017438
 - Luna Campos Emiliano - 320292084
 - Juárez Cruz Joshua - 320124516


Este proyecto implementa y compara tres metaheurísticas para resolver Sudoku:
- **Recocido Simulado (SA)** con enfriamiento no monótono y reheating adaptativo
- **Búsqueda Local Iterada (ILS)** con estrategia de perturbación débil
- **Algoritmo Memético** (híbrido genético + búsqueda local)

---

## Requisitos

- Python 3.7+
- Dependencias: `numpy`, `matplotlib`, `scipy`

Instalar dependencias:
```bash
pip install numpy matplotlib scipy
```

---

## Estructura del Proyecto

```
Tarea5/
├── src/
│   ├── sudoku.py           # Clase base Sudoku y SudokuSolution
│   ├── SA.py               # Recocido Simulado
│   ├── ils.py              # Búsqueda Local Iterada
│   ├── memetico.py         # Algoritmo Memético
│   ├── experimentacion.py  # Orquestador de experimentos
│   ├── graficas.py         # Generación de gráficas
│   └── Ejemplares/         # Archivos de prueba (.txt)
├── output/                 # Resultados por ejemplar (CSV, PKL, PNG)
└── README.md
```

---

## Instrucciones de Ejecución

### Ejercicio 1: Ejecución Individual de Algoritmos

Cada algoritmo puede ejecutarse de forma independiente con un ejemplar específico.

#### Recocido Simulado (SA)
```bash
python3 src/SA.py --ejemplar src/Ejemplares/Easy1.txt --seed 42 --max_iter 1000
```

#### Búsqueda Local Iterada (ILS)
```bash
python3 src/ils.py --ejemplar src/Ejemplares/Medium1.txt --seed 42 --max_iter 1000 --max_iter_local 1000
```

#### Algoritmo Memético
```bash
python3 src/memetico.py --ejemplar src/Ejemplares/Hard1.txt --seed 42 --pop_size 100 --max_iter 100 --local_apply_prob 0.2 --local_max_iter 50
```

**Notas:**
- `--seed` es opcional; si se omite, se genera una semilla aleatoria que se imprime en consola
- Todos los algoritmos imprimen la semilla utilizada para reproducibilidad

---

### Ejercicio 2: Experimentación Completa (30 Repeticiones)

El script `experimentacion.py` ejecuta 30 repeticiones por algoritmo y genera:
- CSV con resultados detallados y estadísticas
- Pickle con datos completos (incluyendo soluciones)
- Gráficas automáticas (evolución, diversidad, entropía, boxplot, etc.)

#### 2.a-d) Ejecutar 30 repeticiones para todos los algoritmos
```bash
python3 src/experimentacion.py --num_ejecuciones 30 --save_poblaciones
```

**Comportamiento:**
- Procesa todos los ejemplares en `src/Ejemplares/`
- Guarda resultados en `output/<Ejemplar>/` (un directorio por ejemplar)
- Incluye semilla en CSV por cada ejecución (columna `Seed`)
- Genera todas las gráficas requeridas automáticamente

#### Ejecutar para un ejemplar específico
```bash
python3 src/experimentacion.py --ejemplar src/Ejemplares/Easy1.txt --num_ejecuciones 30 --save_poblaciones
```

#### Ejecutar solo un algoritmo (30 repeticiones)

**Solo Recocido Simulado:**
```bash
python3 src/experimentacion.py --num_ejecuciones 30 --only_sa --save_poblaciones
```

**Solo ILS:**
```bash
python3 src/experimentacion.py --num_ejecuciones 30 --only_ils --save_poblaciones
```

**Solo Memético:**
```bash
python3 src/experimentacion.py --num_ejecuciones 30 --only_memetico --save_poblaciones
```

**Nota importante:** El flag `--save_poblaciones` habilita:
- Guardado de poblaciones por generación (Memético)
- Guardado de trayectorias por iteración (SA e ILS)
- Generación de gráficas de evolución de entropía por algoritmo

---

### Ejercicio 3: Análisis de Resultados

Todos los entregables se generan automáticamente al ejecutar `experimentacion.py`:

#### 3.a) Tabla de Parámetros
Los parámetros de cada algoritmo están documentados en el CSV generado bajo la sección `RESUMEN COMPARATIVO`.

#### 3.b) Gráficas de Evolución de Aptitud
- **Memético:** `memetico_evolucion_generaciones.png` (aptitud mejor y promedio por generación)
- **SA/ILS:** `evolucion_comparativa_iteraciones.png` (aptitud activa vs mejor-so-far)

#### 3.c) Gráfica de Evolución Promedio
- `1_evolucion_aptitud_todos.png` (comparación de los 3 algoritmos por ejecución)

#### 3.d) Tabla de Resultados Estadísticos
Incluida en el CSV generado:
- Mejor, Peor, Promedio, Mediana, Desviación Estándar
- Número de soluciones óptimas (fitness = 0)

#### 3.e) Gráficas de Diversidad
- `2_diversidad_todos.png` (Hamming y Euclidiana)
- `3_aptitud_diversidad_todos.png` (aptitud + diversidad combinadas)

#### 3.f) Gráficas de Entropía
- `6_entropia.png` (barra comparativa)
- `6_entropia_evolucion_sa.png` (evolución de entropía - SA)
- `6_entropia_evolucion_ils.png` (evolución de entropía - ILS)
- `6_entropia_evolucion_memetico.png` (evolución de entropía - Memético)

#### 3.g) BoxPlot
- `5_boxplot_comparacion.png` (comparación estadística de los 3 algoritmos)

**Ubicación de archivos:**
```
output/
├── Easy1/
│   ├── experimentacion_Easy1_<timestamp>.csv
│   ├── datos_experimentacion_Easy1.pkl
│   ├── 1_evolucion_aptitud_todos.png
│   ├── 2_diversidad_todos.png
│   ├── 3_aptitud_diversidad_todos.png
│   ├── 4_calidad_ejecuciones_todos.png
│   ├── 5_boxplot_comparacion.png
│   ├── 6_entropia.png
│   ├── 6_entropia_evolucion_sa.png
│   ├── 6_entropia_evolucion_ils.png
│   ├── 6_entropia_evolucion_memetico.png
│   ├── evolucion_comparativa_iteraciones.png
│   └── memetico_evolucion_generaciones.png
├── Medium1/
│   └── ... (misma estructura)
└── Hard1/
    └── ... (misma estructura)
```

---

## Ejemplos de Comandos por Inciso

### Inciso 1: Parametrización de Algoritmos
```bash
# Recocido Simulado con enfriamiento no monótono
python3 src/SA.py --ejemplar src/Ejemplares/SD2.txt --max_iter 1000

# ILS con perturbación débil
python3 src/ils.py --ejemplar src/Ejemplares/SD2.txt --max_iter 1000 --max_iter_local 1000

# Memético (híbrido GA + búsqueda local)
python3 src/memetico.py --ejemplar src/Ejemplares/SD2.txt --pop_size 100 --max_iter 100 --local_apply_prob 0.2 --local_max_iter 100
```

### Inciso 2: Experimentación (30 Repeticiones)
```bash
# Todos los algoritmos, todos los ejemplares
python3 src/experimentacion.py --num_ejecuciones 30 --save_poblaciones

# Un ejemplar específico
python3 src/experimentacion.py --ejemplar src/Ejemplares/Easy1.txt --num_ejecuciones 30 --save_poblaciones

# Solo un algoritmo (ejemplo: ILS)
python3 src/experimentacion.py --num_ejecuciones 30 --only_ils --save_poblaciones
```

### Inciso 3: Análisis de Resultados
Los resultados se generan automáticamente en `output/<Ejemplar>/` tras ejecutar el comando del Inciso 2.

**Archivos clave:**
- CSV: estadísticas completas con semillas
- PKL: datos brutos para análisis adicional
- PNG: todas las gráficas requeridas (evolución, diversidad, entropía, boxplot)

---

## Reproducibilidad

Para reproducir una ejecución específica, usar la semilla registrada en el CSV:

```bash
python3 src/SA.py --ejemplar src/Ejemplares/Easy1.txt --seed 1234567890 --max_iter 1000
```

La semilla se imprime en consola y se almacena en:
- Salida estándar del programa
- Columna `Seed` del CSV generado
- Campo `seed` en el archivo PKL

---

## Notas Técnicas

- **Criterio de término justo:** Todos los algoritmos usan `max_iteraciones` (SA/ILS: 1000, Memético: 100 generaciones)
- **Optimización de rendimiento:** Las gráficas de evolución de entropía usan cálculo vectorizado con NumPy
- **Datos pesados:** Usar `--save_poblaciones` aumenta el tamaño de los PKL (necesario para gráficas de evolución de entropía)
