import numpy as np
import csv
from funciones import sphere, ackley, griewank, rastrigin, rosenbrock


def obtener_limites(nombre_funcion):
    '''Retorna los límites según la función'''
    limites = {
        'sphere': (-5.12, 5.12),
        'ackley': (-30, 30),
        'griewank': (-600, 600),
        'rastrigin': (-5.12, 5.12),
        'rosenbrock': (-2.048, 2.048)
    }
    return limites.get(nombre_funcion, (-5.12, 5.12))

def generar_poblacion_inicial(NIND, dim_x, nombre_funcion):  
    limite_inf, limite_sup = obtener_limites(nombre_funcion)
    poblacion = np.random.uniform(limite_inf, limite_sup, size=(NIND, dim_x))
    return poblacion

def cruzar_vectores(x_i, v_i, CR):
    D=len(x_i)
    u_i = np.zeros(D)
    g= np.random.randint(0, D)
    for j in range(D):
        r_j = np.random.rand()
        if r_j < CR or j== g:
            u_i[j]=v_i[j]
        else:
            u_i[j]=x_i[j]
    return u_i

def calcular_vector_mutado(poblacion, i, F):
    N= len(poblacion)
    indices =list (range(N))
    indices.remove(i)
    r1,r2,r3=np.random.choice(indices,3,replace= False)
    x_r1 = poblacion[r1]
    x_r2 = poblacion[r2]
    x_r3 = poblacion[r3]
    v_i= x_r1 + F*(x_r2-x_r3)

    return v_i

def aplicar_limites(individuo, nombre_funcion):
    limite_inf, limite_sup = obtener_limites(nombre_funcion)
    individuo_corregido = individuo.copy()
    mask_superior = individuo > limite_sup
    individuo_corregido[mask_superior] = limite_sup
    mask_inferior = individuo < limite_inf
    individuo_corregido[mask_inferior] = limite_inf
    
    return individuo_corregido


def seleccionar(x_i, u_i, funcion_objetivo):
    fitness_x = funcion_objetivo(x_i)
    fitness_u = funcion_objetivo(u_i)
    
    if fitness_u <= fitness_x:
        return u_i
    else:
        return x_i

def evolucionDif(funcion_objetivo, nombre_funcion, dim_x=10, NIND=50, 
                 max_generaciones=100, F=0.8, CR=0.9, verbose=True):
    poblacion = generar_poblacion_inicial(NIND, dim_x, nombre_funcion)
    historial_fitness = []
    
    for gen in range(max_generaciones):
        nueva_poblacion = np.zeros_like(poblacion) 
        for i in range(NIND):
            x_i = poblacion[i]
            v_i = calcular_vector_mutado(poblacion, i, F)
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
    print("ALGORITMO DE EVOLUCIÓN DIFERENCIAL")
    print("="*60)
    
    funciones = {
        'sphere': sphere,
        'ackley': ackley,
        'griewank': griewank,
        'rastrigin': rastrigin,
        'rosenbrock': rosenbrock
    }
    
    dim_x = 10
    NIND = 50   
    max_gen = 100  
    F = 0.8    
    CR = 0.9  
    
    resultados = []
    
    for nombre, funcion in funciones.items():
        print(f"\n{'='*60}")
        print(f"Optimizando: {nombre.upper()}")
        print(f"{'='*60}")
        
        mejor_solucion, mejor_fitness, historial = evolucionDif(
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
        print(f"Mejor solución: {mejor_solucion}")
        print(f"{'*'*60}\n")
        
        resultados.append({
            'funcion': nombre,
            'mejor_fitness': mejor_fitness,
            'mejor_solucion': mejor_solucion,
            'historial': historial
        })
    
    with open('resultados_evolucion_diferencial.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Función', 'Mejor Fitness'])
        for res in resultados:
            writer.writerow([res['funcion'], res['mejor_fitness']])
    
    print("\n" + "="*60)
    print("Resultados guardados en 'resultados_evolucion_diferencial.csv'")
    print("="*60)


if __name__ == "__main__":
    main()