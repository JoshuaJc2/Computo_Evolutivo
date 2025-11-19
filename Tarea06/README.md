# Tarea 5 
## Cómputo Evolutivo 2026-1

**Equipo:** TheKingdomPinguin
**Integrantes:** 

 - Castillo Hernández Antonio - 320017438
 - Luna Campos Emiliano - 320292084
 - Juárez Cruz Joshua - 320124516


Este proyecto implementa tres variantes de estrategia diferencial y la original y las compara:
- **Original** 
- **Best-1** 
- **Rand-2**
- **Current-to-Best** 

---

## Requisitos

- Python 3.7+
- Dependencias: `numpy`, `matplotlib`, `scipy`, `pandas`

Instalar dependencias:
```bash
pip install numpy matplotlib scipy pandas
```

---

## Estructura del Proyecto

```
Tarea5/
├── src/
│   ├── ED_current_to_best1.py                      # Current-to-best
│   ├── evolucion_diferencial_best1.py              # Best-1
│   ├── evolucion_diferencial_rand2.py              # Rand-2
│   ├── evolucion_diferencial.py                    # Algoritmo base
│   ├── experimentacion.py                          # Orquestador de experimentos
│   ├── graficas.py                                 # Generación de gráficas
│   └── funciones.py                                # Funciones de prueba
│   └── graficas_boxplot/                            # Resultados por ejemplar (CSV, PKL, PNG)
└── README.md
```

---

## Instrucciones de Ejecución

### Ejercicio 1: Ejecución Individual de Algoritmos

Cada algoritmo puede ejecutarse de forma independiente con un ejemplar específico.

#### Rand-1
```bash
python3 src/evolucion_diferencial.py
```

#### Rand-2 
```bash
python3 src/evolucion_diferencial_rand2.py 
```

#### Best-1
```bash
python3 src/evolucion_diferencial_best1.py 
```

#### Current-to-Best
```bash
python3 src/ED_current_to_best1.py 
```
---

### Ejercicio 2: Experimentación Completa (10 Repeticiones)

El script `experimentacion.py` ejecuta 10 repeticiones por algoritmo y valores de F y CR, ademas genera:
- CSV con resultados detallados y estadísticas
- Gráficas boxplot

#### 2.a) Ejecutar 10 repeticiones para todos los algoritmos
```bash
python3 src/experimentacion.py 
```

#### 2.b) Representar los resultados en graficas
```bash
python3 src/graficas.py 
```

**Comportamiento:**
- Guarda resultados en `graficas_boxplot/resumen_estadistico.csv`
- Genera todas las gráficas requeridas automáticamente


### Ejercicio 3: Análisis de Resultados

Todos los entregables se generan automáticamente al ejecutar `experimentacion.py`:

#### 3.a) Tabla de Parámetros
Los parámetros de cada algoritmo están documentados en el CSV generado bajo la sección `RESUMEN ESTADISTICO`.


#### 3.b) BoxPlot
- `DE_best_1__F0.75_CR0.6.png` (comparación estadística de las 5 funciones)
- `DE_best_1__F0.8_CR0.7.png` (comparación estadística de las 5 funciones)
- `DE_best_1__F0.9_CR0.5.png` (comparación estadística de las 5 funciones)
- `DE_current-to-best_1__F0.75_CR0.6.png` (comparación estadística de las 5 funciones)
- `DE_current-to-best_1__F0.8_CR0.7.png` (comparación estadística de las 5 funciones)
- `DE_current-to-best_1__F0.9_CR0.5.png` (comparación estadística de las 5 funciones)
- `DE_rand_1__F0.75_CR0.6.png` (comparación estadística de las 5 funciones)
- `DE_rand_1__F0.8_CR0.7.png` (comparación estadística de las 5 funciones)
- `DE_rand_1__F0.9_CR0.5.png` (comparación estadística de las 5 funciones)
- `DE_rand_2__F0.75_CR0.6.png` (comparación estadística de las 5 funciones)
- `DE_rand_2__F0.8_CR0.7.png` (comparación estadística de las 5 funciones)
- `DE_rand_2__F0.9_CR0.5.png` (comparación estadística de las 5 funciones)


**Ubicación de archivos:**
```
graficas_boxplot/
├
│   ├── DE_best_1__F0.75_CR0.6.png
│   ├── DE_best_1__F0.8_CR0.7.png
│   ├── DE_best_1__F0.9_CR0.5.png
│   ├── DE_current-to-best_1__F0.75_CR0.6.png
│   ├── DE_current-to-best_1__F0.8_CR0.7.png
│   ├── DE_current-to-best_1__F0.9_CR0.5.png
│   ├── DE_rand_1__F0.75_CR0.6.png
│   ├── DE_rand_1__F0.8_CR0.7.png
│   ├── DE_rand_1__F0.9_CR0.5.png
│   ├── DE_rand_2__F0.75_CR0.6.png
│   ├── DE_rand_2__F0.8_CR0.7.png
│   ├── DE_rand_2__F0.9_CR0.5.png
```

---

## Notas Técnicas

- **Criterio de término justo:** Todos los algoritmos usan `max_iteraciones` 300000

