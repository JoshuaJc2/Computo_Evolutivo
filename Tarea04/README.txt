Tarea 4 - Algoritmo Genético vs Busqueda Aleatoria

Integrantes: 
 - Castillo Hernández Antonio - 320017438
 - Luna Campos Emiliano - 320292084
 = Juárez Cruz Joshua - 320124516

Requisitos
- Python 3
- Paquetes: numpy, matplotlib

Instalación de dependencias: 
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

4) Modo interactivo (probar manualmente con diferentes parámetros):
	cd src
	python main.py

	Permite elegir función, selección, cruza, parámetros, etc.


(PD: Si se desea generar diferentes salidas de los algoritmos de acuerdo a lavariante, se puede cambiar los metodos elegidos que quedaron hardcodeados en experimentacion.py y genetico.py o si es solo una prueba unitaria es mejor usar el archivo main.py)
