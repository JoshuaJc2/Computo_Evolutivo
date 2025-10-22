"""
busqueda_aleatoria.py - Baseline para comparación con AG
Genera soluciones aleatorias con el mismo presupuesto de evaluaciones que el AG.
"""

import numpy as np
import csv
from datetime import datetime
from genetico import FUNCIONES

def busqueda_aleatoria(nombre_funcion, dim_x=10, n_bits=16, NIND=100, 
                       max_generaciones=100, mostrar_progreso=True):
    """
    Búsqueda aleatoria pura para baseline.
    Presupuesto: NIND * (max_generaciones + 1) evaluaciones (igual que AG).
    """
    funcion, a, b = FUNCIONES[nombre_funcion.lower()]
    
    # Total de evaluaciones = población inicial + generaciones
    total_evaluaciones = NIND * (max_generaciones + 1)
    
    mejor_fitness = float('inf')
    mejor_solucion = None
    
    if mostrar_progreso:
        print(f"\nBúsqueda Aleatoria: {nombre_funcion} (dim={dim_x})")
        print(f"Presupuesto: {total_evaluaciones} evaluaciones")
    
    # Generar y evaluar soluciones aleatorias
    for eval_num in range(total_evaluaciones):
        # Generar solución aleatoria en el rango
        x = np.random.uniform(a, b, dim_x)
        fitness = funcion(x)
        
        if fitness < mejor_fitness:
            mejor_fitness = fitness
            mejor_solucion = x.copy()
        
        # Mostrar progreso cada 10% del presupuesto
        if mostrar_progreso and (eval_num + 1) % (total_evaluaciones // 10) == 0:
            porcentaje = ((eval_num + 1) / total_evaluaciones) * 100
            print(f"  {porcentaje:.0f}% completado: Mejor fitness = {mejor_fitness:.6f}")
    
    if mostrar_progreso:
        print(f"  Final: Mejor fitness = {mejor_fitness:.6f}\n")
    
    return mejor_fitness, mejor_solucion


def ejecutar_experimentos_aleatorios(num_ejecuciones=10):
    """Ejecuta búsqueda aleatoria 10 veces por función para comparación."""
    params = {
        'dim_x': 10,
        'n_bits': 16,
        'NIND': 100,
        'max_generaciones': 100
    }
    
    funciones = ['sphere', 'ackley', 'griewank', 'rastrigin', 'rosenbrock']
    resultados = []
    
    print("\n" + "="*70)
    print("  BÚSQUEDA ALEATORIA - BASELINE")
    print("="*70)
    
    for func in funciones:
        print(f"\n{func.upper()}:")
        fitness_list = []
        
        for i in range(num_ejecuciones):
            mejor_fitness, mejor_sol = busqueda_aleatoria(
                func, mostrar_progreso=False, **params
            )
            fitness_list.append(mejor_fitness)
            print(f"  Ejecución {i+1:2d}: Fitness = {mejor_fitness:.8f}")
        
        # Estadísticas
        stats = {
            'algoritmo': 'Aleatoria',
            'funcion': func,
            'mejor': np.min(fitness_list),
            'peor': np.max(fitness_list),
            'promedio': np.mean(fitness_list),
            'desviacion': np.std(fitness_list),
            'mediana': np.median(fitness_list),
            'fitness_list': fitness_list
        }
        resultados.append(stats)
        
        print(f"  → Mejor: {stats['mejor']:.8f}, Peor: {stats['peor']:.8f}, "
              f"Promedio: {stats['promedio']:.8f}")
    
    return resultados


def guardar_resultados_csv(resultados):
    """Guarda resultados de búsqueda aleatoria en CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = f'../output/busqueda_aleatoria_{timestamp}.csv'
    
    with open(archivo, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Ejecuciones individuales
        writer.writerow(['Algoritmo', 'Funcion', 'Ejecucion', 'Fitness'])
        for r in resultados:
            for i, fit in enumerate(r['fitness_list'], 1):
                writer.writerow([r['algoritmo'], r['funcion'], i, f"{fit:.10f}"])
        
        writer.writerow([])
        writer.writerow(['ESTADISTICAS RESUMEN'])
        writer.writerow([])
        
        # Resumen estadístico
        writer.writerow(['Algoritmo', 'Funcion', 'Mejor', 'Peor', 'Promedio', 'Mediana', 'Desv.Est.'])
        for r in resultados:
            writer.writerow([
                r['algoritmo'],
                r['funcion'],
                f"{r['mejor']:.10f}",
                f"{r['peor']:.10f}",
                f"{r['promedio']:.10f}",
                f"{r['mediana']:.10f}",
                f"{r['desviacion']:.10f}"
            ])
    
    print(f"\n✓ Resultados guardados: {archivo}")
    return archivo


if __name__ == "__main__":
    resultados = ejecutar_experimentos_aleatorios(10)
    guardar_resultados_csv(resultados)
    
    print("\n" + "="*70)
    print("  COMPLETADO")
    print("="*70)
    print("\nEste baseline debe compararse con los resultados del AG.")
    print("El AG debería superar significativamente estos resultados.\n")
