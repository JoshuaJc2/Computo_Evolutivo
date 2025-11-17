import numpy as np
import csv
import time
from funciones import *
from evolucion_diferencial import calcular_vector_mutado as mutacion_rand1
from evolucion_diferencial_rand2 import calcular_vector_mutado_rand2 as mutacion_rand2
from evolucion_diferencial import generar_poblacion_inicial, cruzar_vectores, aplicar_limites


def mutacion_best1_opt(poblacion, i, F, x_best):
    N = len(poblacion)
    indices = list(range(N))
    indices.remove(i)
    
    # Asegurarnos de no elegir al 'i'
    r2, r3 = np.random.choice(indices, 2, replace=False)
    x_r2 = poblacion[r2]
    x_r3 = poblacion[r3]

    v_i = x_best + F * (x_r2 - x_r3)
    return v_i

def mutacion_current_to_best1_opt(poblacion, i, F, x_best):
    N = len(poblacion)
    indices = list(range(N))
    indices.remove(i)
    
    # Quitar el índice del mejor (si no es 'i')
    best_idx = np.where((poblacion == x_best).all(axis=1))[0]
    if best_idx.size > 0 and best_idx[0] in indices:
        indices.remove(best_idx[0])

    r2, r3 = np.random.choice(indices, 2, replace=False)
    
    x_i = poblacion[i]
    x_r2 = poblacion[r2]
    x_r3 = poblacion[r3]
    
    v_i = x_i + F * (x_best - x_i) + F * (x_r2 - x_r3)
    return v_i



def run_de_experiment(funcion_objetivo, nombre_funcion, variante, 
                        MAX_EVALS, NIND, dim_x, F, CR, verbose=False):
    
    evaluaciones = 0
    historial_fitness = []

    # 1. Población Inicial y Evaluación
    poblacion = generar_poblacion_inicial(NIND, dim_x, nombre_funcion)
    fitness_poblacion = [funcion_objetivo(ind) for ind in poblacion]
    evaluaciones += NIND

    # Guardamos el mejor fitness inicial
    mejor_fitness = np.min(fitness_poblacion)
    historial_fitness.append(mejor_fitness)

    while evaluaciones < MAX_EVALS:
        
        # Encontramos al mejor de la generación ACTUAL (solo 1 vez)
        mejor_idx_gen = np.argmin(fitness_poblacion)
        x_best_gen = poblacion[mejor_idx_gen]

        nueva_poblacion = np.zeros_like(poblacion)
        nuevo_fitness = np.zeros(NIND)

        for i in range(NIND):
            x_i = poblacion[i]
            fitness_x_i = fitness_poblacion[i]

            # --- Selección de Mutación ---
            if variante == 'DE/rand/1':
                v_i = mutacion_rand1(poblacion, i, F)
            elif variante == 'DE/rand/2':
                v_i = mutacion_rand2(poblacion, i, F)
            elif variante == 'DE/best/1':
                v_i = mutacion_best1_opt(poblacion, i, F, x_best_gen)
            elif variante == 'DE/current-to-best/1':
                v_i = mutacion_current_to_best1_opt(poblacion, i, F, x_best_gen)
            
            v_i = aplicar_limites(v_i, nombre_funcion)
            u_i = cruzar_vectores(x_i, v_i, CR)
            u_i = aplicar_limites(u_i, nombre_funcion) 

            
            fitness_u_i = funcion_objetivo(u_i)
            evaluaciones += 1

            if fitness_u_i <= fitness_x_i:
                nueva_poblacion[i] = u_i
                nuevo_fitness[i] = fitness_u_i
            else:
                nueva_poblacion[i] = x_i
                nuevo_fitness[i] = fitness_x_i
            
            
            if evaluaciones >= MAX_EVALS:
                break
        
        poblacion = nueva_poblacion
        fitness_poblacion = nuevo_fitness

        
        mejor_fitness_actual = np.min(fitness_poblacion)
        if mejor_fitness_actual < mejor_fitness:
             mejor_fitness = mejor_fitness_actual
        historial_fitness.append(mejor_fitness)

        if verbose and (evaluaciones % 50000 < NIND):
             print(f"  ...Evals: {evaluaciones}/{MAX_EVALS}, Mejor Fitness: {mejor_fitness:.6f}")

    return mejor_fitness, historial_fitness


def main():
    print("="*60)
    print("INICIANDO EXPERIMENTACIÓN TAREA 6")
    print("="*60)

    FUNCIONES = {'sphere': sphere, 'ackley': ackley,'griewank': griewank,
        'rastrigin': rastrigin,'rosenbrock': rosenbrock}
    
    VARIANTES = {'DE/rand/1': mutacion_rand1, 'DE/rand/2': mutacion_rand2,
    'DE/best/1': mutacion_best1_opt,'DE/current-to-best/1': mutacion_current_to_best1_opt}
    
    
    PARES_PARAMETROS = [(0.9, 0.5), (0.8, 0.7), (0.75, 0.6) ]
    
    N_REPETICIONES = 10 
    MAX_EVALUACIONES = 300000 
    DIMENSION = 10 
    NIND = 50
    
    
    output_filename = 'resultados_experimentacion_total.csv'
    todos_los_resultados = []
    
    start_time = time.time()
    
    print(f"Repeticiones: {N_REPETICIONES}, Evaluaciones: {MAX_EVALUACIONES}")
    print(f"Funciones: {list(FUNCIONES.keys())}")
    print(f"Variantes: {list(VARIANTES.keys())}")
    print(f"Pares (F, CR): {PARES_PARAMETROS}\n")

    
    for func_nombre, func_obj in FUNCIONES.items():
        for var_nombre in VARIANTES.keys():
            for F, CR in PARES_PARAMETROS:
                
                print(f"--- Corriendo: [{func_nombre}] con [{var_nombre}] (F={F}, CR={CR}) ---")
                
                resultados_repeticion = []
                for rep in range(N_REPETICIONES):
                    
                    mejor_fit, historial = run_de_experiment(
                        funcion_objetivo=func_obj,
                        nombre_funcion=func_nombre,
                        variante=var_nombre,
                        MAX_EVALS=MAX_EVALUACIONES,
                        NIND=NIND,
                        dim_x=DIMENSION,
                        F=F,
                        CR=CR,
                        verbose=(rep == 0) 
                    )
                    
                    resultados_repeticion.append(mejor_fit)
                    
                    
                    todos_los_resultados.append({
                        'funcion': func_nombre,
                        'variante': var_nombre,
                        'F': F,
                        'CR': CR,
                        'repeticion': rep + 1,
                        'mejor_fitness': mejor_fit
                    })
                
                # Calcular estadísticas de las 10 repeticiones
                mejor_final = np.min(resultados_repeticion)
                peor_final = np.max(resultados_repeticion)
                media_final = np.mean(resultados_repeticion)
                mediana_final = np.median(resultados_repeticion)
                std_final = np.std(resultados_repeticion)
                
                print(f"  Resultados (10 reps):")
                print(f"    Mejor: {mejor_final:.6f}, Peor: {peor_final:.6f}, Media: {media_final:.6f}")
                print(f"    Mediana: {mediana_final:.6f}, StdDev: {std_final:.6f}\n")

    end_time = time.time()
    print(f"\n" + "="*60)
    print(f"EXPERIMENTO COMPLETADO EN {end_time - start_time:.2f} SEGUNDOS")
    print("="*60)

    try:
        with open(output_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=todos_los_resultados[0].keys())
            writer.writeheader()
            writer.writerows(todos_los_resultados)
        print(f"Resultados detallados guardados en '{output_filename}'")
    except Exception as e:
        print(f"Error al guardar CSV: {e}")

if __name__ == "__main__":
    main()