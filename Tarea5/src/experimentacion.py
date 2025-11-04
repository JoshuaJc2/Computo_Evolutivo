"""
experimentos_sudoku.py - Ejecuta 30 ejecuciones de SA, ILS y Memético para Sudoku
"""

import sys
import numpy as np
import csv
import pickle
from datetime import datetime
import os

from sudoku import Sudoku
from SA import recocido_simulado
from ils import BusquedaLocalIteradaSudoku
from memetico import MemeticGeneticSudoku
import graficas


def ejecutar_recocido_simulado(problema, num_ejecuciones=30):
    """Ejecuta Recocido Simulado múltiples veces"""
    print(f"\n{'='*60}")
    print("  RECOCIDO SIMULADO")
    print("="*60)
    
    resultados = []
    
    params = {
        'temp_inicial': 100.0,
        'alpha': 0.9,
        'max_iteraciones': 1000,
        'max_estancamiento': 650,
        'initial_reheat_factor': 3.0,
        'min_reheat_factor': 1.5,
        'max_evaluaciones': None
    }
    
    for i in range(num_ejecuciones):
        print(f"\nEjecución {i+1}/{num_ejecuciones}")
        print("-" * 40)
        
        mejor, fitness, stats = recocido_simulado(
            problema,
            **params,
            seed=None  # Genera seed aleatoria cada vez
        )
        
        resultado = {
            'ejecucion': i + 1,
            'fitness': fitness,
            'evaluaciones': stats['evaluaciones_totales'],
            'generaciones': stats['generaciones'],
            'tiempo': stats['time'],
            'seed': stats['seed'],
            'solucion': mejor
        }
        resultados.append(resultado)
        
        print(f"  → Fitness: {fitness:.2f}, Evaluaciones: {stats['evaluaciones_totales']}, Tiempo: {stats['time']:.2f}s")
    
    # Estadísticas generales
    fitness_list = [r['fitness'] for r in resultados]
    stats_generales = {
        'algoritmo': 'Recocido Simulado',
        'mejor': np.min(fitness_list),
        'peor': np.max(fitness_list),
        'promedio': np.mean(fitness_list),
        'desviacion': np.std(fitness_list),
        'mediana': np.median(fitness_list),
        'soluciones_optimas': sum(1 for f in fitness_list if f == 0),
        'ejecuciones': resultados,
        'parametros': params
    }
    
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS RECOCIDO SIMULADO:")
    print(f"  Mejor: {stats_generales['mejor']:.2f}")
    print(f"  Peor: {stats_generales['peor']:.2f}")
    print(f"  Promedio: {stats_generales['promedio']:.2f}")
    print(f"  Desviación: {stats_generales['desviacion']:.2f}")
    print(f"  Soluciones óptimas: {stats_generales['soluciones_optimas']}/{num_ejecuciones}")
    
    return stats_generales


def ejecutar_ils(problema, num_ejecuciones=30):
    """Ejecuta Búsqueda Local Iterada múltiples veces"""
    print(f"\n{'='*60}")
    print("  BÚSQUEDA LOCAL ITERADA (ILS)")
    print("="*60)
    
    resultados = []
    
    params = {
        'max_iter': 1000,
        'max_iter_local': 1000
    }
    
    for i in range(num_ejecuciones):
        print(f"\nEjecución {i+1}/{num_ejecuciones}")
        print("-" * 40)
        
        ils = BusquedaLocalIteradaSudoku(problema, seed=None)
        mejor, fitness, stats = ils.iterated_local_search(**params)
        
        resultado = {
            'ejecucion': i + 1,
            'fitness': fitness,
            'evaluaciones': stats['evaluaciones_totales'],
            'iteraciones': stats['iteraciones_totales'],
            'seed': stats['seed'],
            'solucion': mejor
        }
        resultados.append(resultado)
        
        print(f"  → Fitness: {fitness:.2f}, Evaluaciones: {stats['evaluaciones_totales']}")
    
    # Estadísticas generales
    fitness_list = [r['fitness'] for r in resultados]
    stats_generales = {
        'algoritmo': 'ILS',
        'mejor': np.min(fitness_list),
        'peor': np.max(fitness_list),
        'promedio': np.mean(fitness_list),
        'desviacion': np.std(fitness_list),
        'mediana': np.median(fitness_list),
        'soluciones_optimas': sum(1 for f in fitness_list if f == 0),
        'ejecuciones': resultados,
        'parametros': params
    }
    
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS ILS:")
    print(f"  Mejor: {stats_generales['mejor']:.2f}")
    print(f"  Peor: {stats_generales['peor']:.2f}")
    print(f"  Promedio: {stats_generales['promedio']:.2f}")
    print(f"  Desviación: {stats_generales['desviacion']:.2f}")
    print(f"  Soluciones óptimas: {stats_generales['soluciones_optimas']}/{num_ejecuciones}")
    
    return stats_generales


def ejecutar_memetico(problema, num_ejecuciones=30):
    """Ejecuta Algoritmo Genético Memético múltiples veces"""
    print(f"\n{'='*60}")
    print("  ALGORITMO GENÉTICO MEMÉTICO")
    print("="*60)
    
    resultados = []
    
    params = {
        'pop_size': 100,
        'p_cruce': 0.8,
        'p_mut': 0.01,
        'torneo_k': 3,
        'max_generaciones': 100,
        'local_apply_prob': 0.2,
        'local_apply_top': 0,
        'local_max_iter': 50
    }
    
    for i in range(num_ejecuciones):
        print(f"\nEjecución {i+1}/{num_ejecuciones}")
        print("-" * 40)
        
        ga_mem = MemeticGeneticSudoku(
            problema,
            **params,
            seed=None
        )
        
        mejor, fitness, stats = ga_mem.run(verbose=False)
        
        resultado = {
            'ejecucion': i + 1,
            'fitness': fitness,
            'evaluaciones': stats['evaluaciones_totales'],
            'generaciones': stats['generaciones'],
            'solucion': mejor
        }
        resultados.append(resultado)
        
        print(f"  → Fitness: {fitness:.2f}, Evaluaciones: {stats['evaluaciones_totales']}, Generaciones: {stats['generaciones']}")
    
    # Estadísticas generales
    fitness_list = [r['fitness'] for r in resultados]
    stats_generales = {
        'algoritmo': 'Memético',
        'mejor': np.min(fitness_list),
        'peor': np.max(fitness_list),
        'promedio': np.mean(fitness_list),
        'desviacion': np.std(fitness_list),
        'mediana': np.median(fitness_list),
        'soluciones_optimas': sum(1 for f in fitness_list if f == 0),
        'ejecuciones': resultados,
        'parametros': params
    }
    
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS MEMÉTICO:")
    print(f"  Mejor: {stats_generales['mejor']:.2f}")
    print(f"  Peor: {stats_generales['peor']:.2f}")
    print(f"  Promedio: {stats_generales['promedio']:.2f}")
    print(f"  Desviación: {stats_generales['desviacion']:.2f}")
    print(f"  Soluciones óptimas: {stats_generales['soluciones_optimas']}/{num_ejecuciones}")
    
    return stats_generales


def guardar_resultados_csv(resultados_sa, resultados_ils, resultados_mem, ejemplar_name=None, output_dir=None):
    """Guarda todos los resultados en CSV.
    Si ejemplar_name se proporciona y/o output_dir se especifica, guarda en esa carpeta.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determinar carpeta de salida
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_dir = output_dir
    else:
        os.makedirs('../output', exist_ok=True)
        base_dir = '../output'

    tag = f"_{ejemplar_name}" if ejemplar_name else ""
    # sanear tag (quitar barras)
    tag = tag.replace('/', '_').replace('\\', '_')
    archivo = os.path.join(base_dir, f'experimentacion{tag}_{timestamp}.csv')
    
    with open(archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Encabezado
        writer.writerow(['RESULTADOS EXPERIMENTALES - SUDOKU'])
        writer.writerow([f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([])
        
        # Resultados detallados de SA
        writer.writerow(['RECOCIDO SIMULADO - Ejecuciones Individuales'])
        writer.writerow(['Ejecucion', 'Fitness', 'Evaluaciones', 'Generaciones', 'Tiempo(s)', 'Seed'])
        for r in resultados_sa['ejecuciones']:
            writer.writerow([
                r['ejecucion'],
                f"{r['fitness']:.2f}",
                r['evaluaciones'],
                r['generaciones'],
                f"{r['tiempo']:.2f}",
                r['seed']
            ])
        writer.writerow([])
        
        # Resultados detallados de ILS
        writer.writerow(['BÚSQUEDA LOCAL ITERADA - Ejecuciones Individuales'])
        writer.writerow(['Ejecucion', 'Fitness', 'Evaluaciones', 'Iteraciones', 'Seed'])
        for r in resultados_ils['ejecuciones']:
            writer.writerow([
                r['ejecucion'],
                f"{r['fitness']:.2f}",
                r['evaluaciones'],
                r['iteraciones'],
                r['seed']
            ])
        writer.writerow([])
        
        # Resultados detallados de Memético
        writer.writerow(['ALGORITMO MEMÉTICO - Ejecuciones Individuales'])
        writer.writerow(['Ejecucion', 'Fitness', 'Evaluaciones', 'Generaciones'])
        for r in resultados_mem['ejecuciones']:
            writer.writerow([
                r['ejecucion'],
                f"{r['fitness']:.2f}",
                r['evaluaciones'],
                r['generaciones']
            ])
        writer.writerow([])
        
        # Resumen comparativo
        writer.writerow(['RESUMEN COMPARATIVO'])
        writer.writerow(['Algoritmo', 'Mejor', 'Peor', 'Promedio', 'Mediana', 'Desv.Est.', 'Óptimas'])
        
        for resultado in [resultados_sa, resultados_ils, resultados_mem]:
            writer.writerow([
                resultado['algoritmo'],
                f"{resultado['mejor']:.2f}",
                f"{resultado['peor']:.2f}",
                f"{resultado['promedio']:.2f}",
                f"{resultado['mediana']:.2f}",
                f"{resultado['desviacion']:.2f}",
                resultado['soluciones_optimas']
            ])
    
    print(f"\n✓ Resultados guardados en: {archivo}")
    return archivo


def guardar_datos_pickle(resultados_sa, resultados_ils, resultados_mem, ejemplar_name=None, output_dir=None):
    """Guarda datos completos en pickle para análisis posterior"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_dir = output_dir
    else:
        os.makedirs('../output', exist_ok=True)
        base_dir = '../output'

    datos = {
        'recocido_simulado': resultados_sa,
        'ils': resultados_ils,
        'memetico': resultados_mem,
        'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tag = f"_{ejemplar_name}" if ejemplar_name else ""
    tag = tag.replace('/', '_').replace('\\', '_')
    archivo = os.path.join(base_dir, f'datos_experimentacion{tag}.pkl')
    with open(archivo, 'wb') as f:
        pickle.dump(datos, f)

    print(f"✓ Datos completos guardados en: {archivo}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ejecutar experimentos con SA, ILS y Memético para Sudoku')
    parser.add_argument('--ejemplar', type=str, default=None, 
                        help='Ruta al archivo del ejemplar de Sudoku')
    parser.add_argument('--num_ejecuciones', type=int, default=30, 
                        help='Número de ejecuciones por algoritmo (default: 30)')
    args = parser.parse_args()
    
    # Cargar problema(s)
    script_dir = os.path.dirname(__file__)
    ejemplares_dir = os.path.join(script_dir, 'Ejemplares')

    if args.ejemplar:
        problema = Sudoku.from_file(args.ejemplar)
        ejemplar_name = os.path.splitext(os.path.basename(args.ejemplar))[0]
        ejemplares_to_run = [(ejemplar_name, problema)]
        print(f"Ejemplar cargado desde: {args.ejemplar}")
    elif os.path.isdir(ejemplares_dir):
        # Recorrer todos los archivos en Ejemplares/
        filenames = sorted([f for f in os.listdir(ejemplares_dir) if os.path.isfile(os.path.join(ejemplares_dir, f))])
        ejemplares_to_run = []
        for fn in filenames:
            path = os.path.join(ejemplares_dir, fn)
            try:
                problema = Sudoku.from_file(path)
                ejemplar_name = os.path.splitext(fn)[0]
                ejemplares_to_run.append((ejemplar_name, problema))
                print(f"Añadido ejemplar: {fn}")
            except Exception as e:
                print(f"Omitiendo {fn}: no se pudo cargar ({e})")
    else:
        # Grid de ejemplo
        grid = [
            [1, 0, 0, 0, 0, 7, 0, 9, 0],
            [0, 3, 0, 0, 2, 0, 0, 0, 8],
            [0, 0, 9, 6, 0, 0, 5, 0, 0],
            [0, 0, 5, 3, 0, 0, 9, 0, 0],
            [0, 1, 0, 0, 8, 0, 0, 0, 2],
            [6, 0, 0, 0, 0, 4, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 7, 0, 0, 0, 3, 0, 0]
        ]
        problema = Sudoku(grid)
        ejemplares_to_run = [('ejemplo', problema)]
        print("Usando grid de ejemplo")

    # Ejecutar experimentos para cada ejemplar detectado
    for ejemplar_name, problema in ejemplares_to_run:
        print(f"\nTablero inicial ({ejemplar_name}):")
        print(problema.grid)
        print(f"\nEjecutando {args.num_ejecuciones} ejecuciones de cada algoritmo para: {ejemplar_name}...")

        resultados_sa = ejecutar_recocido_simulado(problema, args.num_ejecuciones)
        resultados_ils = ejecutar_ils(problema, args.num_ejecuciones)
        resultados_mem = ejecutar_memetico(problema, args.num_ejecuciones)

        # Guardar resultados para este ejemplar (incluye nombre en el archivo)
        print(f"\n{'='*60}")
        print(f"GUARDANDO RESULTADOS ({ejemplar_name})")
        print("="*60)

        # Calcular ruta absoluta de salida relativa al directorio del script.
        # Esto evita que ejecutar `python3 src/experimentacion.py` desde la raíz
        # coloque los resultados en `../output` relativo al CWD.
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        out_dir = os.path.join(project_root, 'output', ejemplar_name)
        os.makedirs(out_dir, exist_ok=True)

        # Guardar archivos (CSV y pickle) dentro de la subcarpeta del ejemplar
        guardar_resultados_csv(resultados_sa, resultados_ils, resultados_mem, ejemplar_name=ejemplar_name, output_dir=out_dir)
        guardar_datos_pickle(resultados_sa, resultados_ils, resultados_mem, ejemplar_name=ejemplar_name, output_dir=out_dir)

        # Generar gráficas en la misma carpeta del ejemplar
        try:
            graficas.graficar_evolucion_aptitud_todos(resultados_sa, resultados_ils, resultados_mem, out_dir)
            graficas.graficar_diversidad_todos(resultados_sa, resultados_ils, resultados_mem, out_dir)
            graficas.graficar_aptitud_diversidad_todos(resultados_sa, resultados_ils, resultados_mem, out_dir)
            graficas.graficar_calidad_ejecuciones_todos(resultados_sa, resultados_ils, resultados_mem, out_dir)
            graficas.graficar_boxplot_comparacion(resultados_sa, resultados_ils, resultados_mem, out_dir)
            graficas.graficar_entropia(resultados_sa, resultados_ils, resultados_mem, out_dir)
        except Exception as e:
            print(f"Error generando gráficas para {ejemplar_name}: {e}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print("="*60)
    print(f"\n{'Algoritmo':<20} {'Mejor':<10} {'Promedio':<10} {'Óptimas':<10}")
    print("-" * 60)
    print(f"{'Recocido Simulado':<20} {resultados_sa['mejor']:<10.2f} {resultados_sa['promedio']:<10.2f} {resultados_sa['soluciones_optimas']:<10}")
    print(f"{'ILS':<20} {resultados_ils['mejor']:<10.2f} {resultados_ils['promedio']:<10.2f} {resultados_ils['soluciones_optimas']:<10}")
    print(f"{'Memético':<20} {resultados_mem['mejor']:<10.2f} {resultados_mem['promedio']:<10.2f} {resultados_mem['soluciones_optimas']:<10}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENTOS COMPLETADOS")
    print("="*60)
