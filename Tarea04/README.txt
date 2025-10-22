Tarea 4 - Algoritmo Genético (Optimización continua)

Requisitos
- Python 3
- Paquetes: numpy, matplotlib

Instalación de dependencias (opcional)
	pip install numpy matplotlib

Ejecuciones y generación de salidas
1) Ejecutar experimentos (10 corridas por función, dim=10):
	cd src
	python experimentacion.py

	Esto genera:
	- output/ejecuciones_YYYYMMDD_HHMMSS.csv (estadísticas y corridas)
	- output/evolucion_promedio_<func>_YYYYMMDD_HHMMSS.csv
	- output/datos_experimentos.pkl

2) Generar gráficas consolidadas:
	cd src
	python graficas.py

	Esto genera en output/:
	- 1_evolucion_aptitud_todas.png
	- 2_evolucion_diversidad_todas.png
	- 3_aptitud_diversidad_todas.png
	- 4_calidad_ejecuciones_todas.png
	- 5_boxplot_comparacion.png
	- evolucion_<func>_mejor_vs_promedio.png (una por función)

Notas de parámetros relevantes (predeterminados):
- dim_x=10, n_bits=16, NIND=100
- max_generaciones=100, porcNewInd=0.8 (reemplazo de peores por porcentaje)
- funSeleccion='ruleta' (minimización vía transformación 1)
- tipoCruza='uniforme', probCruza=0.8 (clona si no cruza)
- probMutacion=0.01 (flip bit por bit)

3) Ejecutar búsqueda aleatoria (baseline para comparación):
	cd src
	python busqueda_aleatoria.py

	Esto genera:
	- output/busqueda_aleatoria_YYYYMMDD_HHMMSS.csv

4) Ejecutar variantes del AG (inciso 2.b):
	cd src
	python experimentacion_variantes.py

	Ejecuta 4 variantes × 5 funciones × 10 corridas
	Genera: output/comparativa_variantes_YYYYMMDD_HHMMSS.csv

5) Modo interactivo (probar manualmente con diferentes parámetros):
	cd src
	python main.py

	Permite elegir función, selección, cruza, parámetros, etc.

Ejecutar un ejemplo rápido (solo una función):
	cd src
	python -c "from genetico import algoritmo_genetico; algoritmo_genetico('sphere')"
