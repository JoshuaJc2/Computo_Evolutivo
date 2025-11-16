import numpy as np
import csv
from funciones import*
from evolucion_diferencial import*

def calcular_vector_mutado_best1(poblacion, i, F, funcion_objetivo):
    N = len(poblacion)
    indices = list(range(N))
    indices.remove(i)
    
    fitness=  [funcion_objetivo(ind) for ind in poblacion]
    indice_mejor = np.argmin(fitness)
    x_best = poblacion[indice_mejor]


    r2, r3= np.random.choice(indices, 2, replace=False)
    x_r2 = poblacion[r2]
    x_r3 = poblacion[r3]

    v_i = x_best + F*(x_r2-x_r3)

    return v_i


def evolucionDif_best1(funcion_objetivo, nombre_funcion, dim_x=10, NIND=50, 
                       max_generaciones=100, F=0.5, CR=0.8, verbose=True):
    
    poblacion = generar_poblacion_inicial(NIND, dim_x, nombre_funcion)
    historial_fitness = []
    
    for gen in range(max_generaciones):
        nueva_poblacion = np.zeros_like(poblacion) 
        for i in range(NIND):
            x_i = poblacion[i]
            v_i = calcular_vector_mutado_best1(poblacion, i, F, funcion_objetivo)
            v_i = aplicar_limites(v_i, nombre_funcion)
            u_i = cruzar_vectores(x_i, v_i, CR)
            u_i = aplicar_limites(u_i, nombre_funcion)
            nueva_poblacion[i] = seleccionar(x_i, u_i, funcion_objetivo)
        
        poblacion = nueva_poblacion
        
        fitness_valores = [funcion_objetivo(ind) for ind in poblacion]
        mejor_fitness = np.min(fitness_valores)
        historial_fitness.append(mejor_fitness)
        
        if verbose and (gen % 10 == 0 or gen == max_generaciones - 1):
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness:.6f}")
    
    fitness_finales = [funcion_objetivo(ind) for ind in poblacion]
    mejor_indice = np.argmin(fitness_finales)
    
    return poblacion[mejor_indice], fitness_finales[mejor_indice], historial_fitness


def main():
    print("="*60)
    print("ALGORITMO DE EVOLUCIÓN DIFERENCIAL (Variante: DE/best/1)")
    print("="*60)
    
    funciones = {
        'sphere': sphere,
        'ackley': ackley,
        'griewank': griewank,
        'rastrigin': rastrigin,
        'rosenbrock': rosenbrock
    }
    
    # Parámetros del algoritmo
    dim_x = 10
    NIND = 50   
    max_gen = 100  
    F = 0.8    
    CR = 0.9
    
    print(f"\nParámetros:")
    print(f"  Dimensión: {dim_x}")
    print(f"  Población: {NIND}")
    print(f"  Generaciones: {max_gen}")
    print(f"  F (Mutación): {F}")
    print(f"  CR (Cruce): {CR}")
    print(f"  Estrategia: DE/best/1\n")
    
    resultados = []
    
    for nombre, funcion in funciones.items():
        print(f"\n{'='*60}")
        print(f"Optimizando: {nombre.upper()}")
        print(f"{'='*60}")
        
        mejor_solucion, mejor_fitness, historial = evolucionDif_best1( 
            funcion_objetivo=funcion,
            nombre_funcion=nombre,
            dim_x=dim_x,
            NIND=NIND, 
            max_generaciones=max_gen,
            F=F,
            CR=CR,
            verbose=True
        )
        
        print(f"\n{'*'*60}")
        print(f"RESULTADOS FINALES - {nombre.upper()}")
        print(f"{'*'*60}")
        print(f"Mejor fitness: {mejor_fitness:.10f}")
        print(f"Mejor solución (primeros 5 valores): {mejor_solucion[:5]}")
        print(f"{'*'*60}\n")
        
        resultados.append({
            'funcion': nombre,
            'mejor_fitness': mejor_fitness,
            'mejor_solucion': mejor_solucion,
            'historial': historial
        })
    
    # Guardar resultados
    output_filename = 'resultados_evolucion_diferencial_best1.csv'
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Función', 'Mejor Fitness'])
        for res in resultados:
            writer.writerow([res['funcion'], res['mejor_fitness']])
    
    print("\n" + "="*60)
    print(f"Resultados guardados en '{output_filename}'")
    print("="*60)


if __name__ == "__main__":
    main()