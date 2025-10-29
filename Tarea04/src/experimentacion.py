"""
experimentos.py - Ejecuta 10+ veces cada función con tracking de diversidad
"""

import sys
import numpy as np
import csv
import pickle
from datetime import datetime

from genetico import generar_poblacion_inicial, evaluar_poblacion, generar_nueva_poblacion, FUNCIONES
from codificacion import decodifica_array


def busqueda_aleatoria_simple(nombre_funcion, dim_x=10, NIND=100, max_generaciones=100):
    """
    Búsqueda aleatoria para baseline (Inciso 2.a).
    Mismo presupuesto que AG: NIND * (max_generaciones + 1) evaluaciones.
    """
    funcion, a, b = FUNCIONES[nombre_funcion.lower()]
    total_evaluaciones = NIND * (max_generaciones + 1)
    
    mejor_fitness = float('inf')
    mejor_solucion = None
    
    for _ in range(total_evaluaciones):
        x = np.random.uniform(a, b, dim_x)
        fitness = funcion(x)
        if fitness < mejor_fitness:
            mejor_fitness = fitness
            mejor_solucion = x.copy()
    
    return {'mejor_fitness': mejor_fitness, 'mejor_solucion': mejor_solucion}


def distancia_hamming(ind1, ind2):
    return np.sum(ind1 != ind2)


def calcular_diversidad_poblacion(poblacion):
    """
    Calcula la diversidad promedio de la población usando distancia de Hamming.
    Diversidad = promedio de distancias entre todos los pares de individuos
    """
    n = len(poblacion)
    if n < 2:
        return 0.0
    
    distancias = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = distancia_hamming(poblacion[i], poblacion[j])
            distancias.append(dist)
    
    return np.mean(distancias)


def calcular_diversidad_normalizada(poblacion):
    """
    Diversidad normalizada (0 a 1).
    Normaliza por la longitud del cromosoma.
    """
    diversidad = calcular_diversidad_poblacion(poblacion)
    longitud_cromosoma = len(poblacion[0])
    return diversidad / longitud_cromosoma if longitud_cromosoma > 0 else 0.0


def algoritmo_con_tracking(nombre_funcion, dim_x=10, n_bits=16, NIND=100, 
                           max_generaciones=100, porcNewInd=0.8, probMutacion=0.01,
                           funSeleccion='ruleta', funCruza='cruza_un_punto'):
    """Ejecuta el AG con tracking de aptitud Y diversidad"""
    funcion, a, b = FUNCIONES[nombre_funcion.lower()]
    poblacion = generar_poblacion_inicial(NIND, dim_x, n_bits)
    fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
    
    # Tracking de aptitud
    hist_mejor = [np.min(fitness)]
    hist_promedio = [np.mean(fitness)]
    hist_peor = [np.max(fitness)]
    
    # Tracking de diversidad
    hist_diversidad = [calcular_diversidad_poblacion(poblacion)]
    hist_diversidad_norm = [calcular_diversidad_normalizada(poblacion)]
    
    mejor_fitness = np.min(fitness)
    mejor_individuo = poblacion[np.argmin(fitness)].copy()
    
    # Evolución
    for gen in range(1, max_generaciones + 1):
        poblacion = generar_nueva_poblacion(poblacion, fitness, porcNewInd, probMutacion, funSeleccion, funCruza)
        fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
        
        # Guardar aptitud
        hist_mejor.append(np.min(fitness))
        hist_promedio.append(np.mean(fitness))
        hist_peor.append(np.max(fitness))
        
        # Guardar diversidad
        hist_diversidad.append(calcular_diversidad_poblacion(poblacion))
        hist_diversidad_norm.append(calcular_diversidad_normalizada(poblacion))
        
        if np.min(fitness) < mejor_fitness:
            mejor_fitness = np.min(fitness)
            mejor_individuo = poblacion[np.argmin(fitness)].copy()
    
    mejor_x = decodifica_array(mejor_individuo, dim_x, n_bits, a, b)
    
    return {
        'mejor_fitness': mejor_fitness,
        'mejor_solucion': mejor_x,
        'hist_mejor': hist_mejor,
        'hist_promedio': hist_promedio,
        'hist_peor': hist_peor,
        'hist_diversidad': hist_diversidad,
        'hist_diversidad_norm': hist_diversidad_norm
    }


def ejecutar_experimentos(num_ejecuciones=10):
    """Ejecuta 10+ veces cada función y guarda resultados"""
    
    # Parámetros
    params = {
        'dim_x': 10,
        'n_bits': 16,
        'NIND': 100,
        'max_generaciones': 100,
        'porcNewInd': 0.8,
        'probMutacion': 0.01,
        'funSeleccion': 'elitismo',
        'funCruza': 'cruza_uniforme'
    }
    
    funciones = ['sphere', 'ackley', 'griewank', 'rastrigin', 'rosenbrock']
    resultados = []
    
    print(f"\nEjecutando {num_ejecuciones} veces cada función...")
    print("="*60)
    
    for func in funciones:
        print(f"\n{func.upper()}:")
        fitness_list = []
        ejecuciones = []
        
        for i in range(num_ejecuciones):
            resultado = algoritmo_con_tracking(func, **params)
            fitness_list.append(resultado['mejor_fitness'])
            ejecuciones.append(resultado)
            print(f"  Ejecución {i+1:2d}: Fitness={resultado['mejor_fitness']:.8f}, "
                  f"Div.Final={resultado['hist_diversidad_norm'][-1]:.4f}")
        
        # Estadísticas
        stats = {
            'funcion': func,
            'mejor': np.min(fitness_list),
            'peor': np.max(fitness_list),
            'promedio': np.mean(fitness_list),
            'desviacion': np.std(fitness_list),
            'mediana': np.median(fitness_list),
            'fitness_list': fitness_list,
            'ejecuciones': ejecuciones,
            'parametros': params
        }
        resultados.append(stats)
        
        print(f"  → Mejor: {stats['mejor']:.8f}, Peor: {stats['peor']:.8f}, "
              f"Promedio: {stats['promedio']:.8f}")
    
    return resultados


def guardar_csv(resultados):
    """Guarda resultados en CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = f'../output/ejecuciones_{timestamp}.csv'
    
    with open(archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Datos de cada ejecución
        writer.writerow(['Funcion', 'Ejecucion', 'Fitness', 'Diversidad_Final'])
        for r in resultados:
            for i, (fit, ejec) in enumerate(zip(r['fitness_list'], r['ejecuciones']), 1):
                div_final = ejec['hist_diversidad_norm'][-1]
                writer.writerow([r['funcion'], i, f"{fit:.10f}", f"{div_final:.6f}"])
        
        # Separador
        writer.writerow([])
        writer.writerow(['ESTADISTICAS RESUMEN'])
        writer.writerow([])
        
        # Resumen
        writer.writerow(['Funcion', 'Mejor', 'Peor', 'Promedio', 'Mediana', 'Desv.Est.'])
        for r in resultados:
            writer.writerow([
                r['funcion'],
                f"{r['mejor']:.10f}",
                f"{r['peor']:.10f}",
                f"{r['promedio']:.10f}",
                f"{r['mediana']:.10f}",
                f"{r['desviacion']:.10f}"
            ])
    
    print(f"\n CSV guardado: {archivo}")


def exportar_datos_graficas(resultados):
    """
    Exporta datos específicos para cada tipo de gráfica.
    Crea archivos CSV separados para facilitar el análisis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Para cada función, exportar evolución promedio
    for r in resultados:
        func = r['funcion']
        num_ejecuciones = len(r['ejecuciones'])
        num_generaciones = len(r['ejecuciones'][0]['hist_mejor'])
        
        # Promediar sobre todas las ejecuciones
        mejor_prom = np.zeros(num_generaciones)
        promedio_prom = np.zeros(num_generaciones)
        peor_prom = np.zeros(num_generaciones)
        div_prom = np.zeros(num_generaciones)
        div_norm_prom = np.zeros(num_generaciones)
        
        for ejec in r['ejecuciones']:
            mejor_prom += np.array(ejec['hist_mejor'])
            promedio_prom += np.array(ejec['hist_promedio'])
            peor_prom += np.array(ejec['hist_peor'])
            div_prom += np.array(ejec['hist_diversidad'])
            div_norm_prom += np.array(ejec['hist_diversidad_norm'])
        
        mejor_prom /= num_ejecuciones
        promedio_prom /= num_ejecuciones
        peor_prom /= num_ejecuciones
        div_prom /= num_ejecuciones
        div_norm_prom /= num_ejecuciones
        
        # Guardar en CSV
        archivo = f'../output/evolucion_promedio_{func}_{timestamp}.csv'
        with open(archivo, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generacion', 'Mejor', 'Promedio', 'Peor', 'Diversidad', 'Diversidad_Norm'])
            for gen in range(num_generaciones):
                writer.writerow([
                    gen,
                    f"{mejor_prom[gen]:.10f}",
                    f"{promedio_prom[gen]:.10f}",
                    f"{peor_prom[gen]:.10f}",
                    f"{div_prom[gen]:.6f}",
                    f"{div_norm_prom[gen]:.6f}"
                ])
        
        print(f" Datos para gráficas: evolucion_promedio_{func}_{timestamp}.csv")


def guardar_datos(resultados):
    """Guarda datos para graficas.py"""
    with open('../output/datos_experimentos.pkl', 'wb') as f:
        pickle.dump(resultados, f)
    print(" Datos guardados: ../output/datos_experimentos.pkl")


def ejecutar_busqueda_aleatoria(num_ejecuciones=10):
    """Ejecuta búsqueda aleatoria para comparación (Inciso 2.a)."""
    params = {'dim_x': 10, 'NIND': 100, 'max_generaciones': 100}
    funciones = ['sphere', 'ackley', 'griewank', 'rastrigin', 'rosenbrock']
    resultados = []
    
    print(f"\n{'='*60}")
    print("  BÚSQUEDA ALEATORIA (Baseline)")
    print("="*60)
    
    for func in funciones:
        print(f"\n{func.upper()}:")
        fitness_list = []
        for i in range(num_ejecuciones):
            resultado = busqueda_aleatoria_simple(func, **params)
            fitness_list.append(resultado['mejor_fitness'])
            print(f"  Ejecución {i+1:2d}: Fitness={resultado['mejor_fitness']:.8f}")
        
        stats = {
            'funcion': func,
            'algoritmo': 'BusquedaAleatoria',
            'mejor': np.min(fitness_list),
            'peor': np.max(fitness_list),
            'promedio': np.mean(fitness_list),
            'desviacion': np.std(fitness_list),
            'mediana': np.median(fitness_list),
            'fitness_list': fitness_list
        }
        resultados.append(stats)
        print(f"  → Mejor: {stats['mejor']:.8f}, Promedio: {stats['promedio']:.8f}")
    
    return resultados


if __name__ == "__main__":
    num_ejecuciones = 10
    if len(sys.argv) > 1:
        num_ejecuciones = int(sys.argv[1])
    
    # Ejecutar AG
    resultados_ag = ejecutar_experimentos(num_ejecuciones)
    guardar_csv(resultados_ag)
    exportar_datos_graficas(resultados_ag)
    guardar_datos(resultados_ag)
    
    # Ejecutar búsqueda aleatoria para comparación
    print("\n\n")
    resultados_aleatoria = ejecutar_busqueda_aleatoria(num_ejecuciones)
    
    # Guardar comparativa
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_comp = f'../output/comparativa_ag_vs_aleatoria_{timestamp}.csv'
    with open(archivo_comp, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algoritmo', 'Funcion', 'Mejor', 'Peor', 'Promedio', 'Mediana', 'Desv.Est.'])
        for r in resultados_ag:
            writer.writerow(['AG', r['funcion'], f"{r['mejor']:.10f}", 
                           f"{r['peor']:.10f}", f"{r['promedio']:.10f}",
                           f"{r['mediana']:.10f}", f"{r['desviacion']:.10f}"])
        for r in resultados_aleatoria:
            writer.writerow([r['algoritmo'], r['funcion'], f"{r['mejor']:.10f}",
                           f"{r['peor']:.10f}", f"{r['promedio']:.10f}",
                           f"{r['mediana']:.10f}", f"{r['desviacion']:.10f}"])
    print(f"\n✓ Comparativa guardada: {archivo_comp}")
    
    print("\n" + "="*60)
    print(" COMPLETADO")
    print("="*60)
    print("\nAhora ejecuta: python graficas.py\n")