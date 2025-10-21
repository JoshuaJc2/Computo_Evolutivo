import sys
import numpy as np
from genetico import algoritmo_genetico, FUNCIONES

def pedir_float(mensaje, default):
    try:
        val = input(f"{mensaje} [{default}]: ")
        return float(val) if val.strip() else default
    except Exception:
        return default

def pedir_int(mensaje, default):
    try:
        val = input(f"{mensaje} [{default}]: ")
        return int(val) if val.strip() else default
    except Exception:
        return default

def main():
    print("\n--- Algoritmo Genético Interactivo ---\n")
    print("Funciones disponibles:")
    for i, f in enumerate(FUNCIONES.keys()):
        print(f"  {i+1}) {f}")
    idx = pedir_int("Selecciona la función a optimizar (número)", 1) - 1
    nombre_funcion = list(FUNCIONES.keys())[idx % len(FUNCIONES)]

    dim_x = pedir_int("Dimensión del problema", 10)
    n_bits = pedir_int("Bits por variable", 16)
    NIND = pedir_int("Tamaño de la población", 100)
    max_generaciones = pedir_int("Número de generaciones", 100)
    porcNewInd = pedir_float("Porcentaje de reemplazo de peores (0-1)", 0.8)
    probMutacion = pedir_float("Probabilidad de mutación por bit (0-1)", 0.01)
    probCruza = pedir_float("Probabilidad de cruza (0-1)", 0.8)

    print("\nMétodos de selección:")
    print("  1) ruleta\n  2) torneo\n  3) elitismo")
    sel_idx = pedir_int("Selecciona método de selección", 1)
    funSeleccion = ['ruleta', 'torneo', 'elitismo'][(sel_idx-1)%3]

    print("\nTipos de cruza:")
    print("  1) uniforme\n  2) un_punto")
    cruza_idx = pedir_int("Selecciona tipo de cruza", 1)
    tipoCruza = ['uniforme', 'un_punto'][(cruza_idx-1)%2]

    print("\nEjecutando algoritmo genético...")
    # Llama el algoritmo con los parámetros elegidos
    if tipoCruza == 'uniforme':
        # Usamos la versión con cruza uniforme
        from genetico import generar_nueva_poblacion, generar_poblacion_inicial, evaluar_poblacion
        from codificacion import decodifica_array
        print(f"Optimizando {nombre_funcion} con dimensión {dim_x}")
        ind, fit, sol = None, None, None
        funcion, a, b = FUNCIONES[nombre_funcion]
        poblacion = generar_poblacion_inicial(NIND, dim_x, n_bits)
        fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
        mejor_fitness_historico = np.min(fitness)
        mejor_individuo = poblacion[np.argmin(fitness)].copy()
        print(f"Generación 0: Mejor fitness = {mejor_fitness_historico:.6f}")
        for gen in range(1, max_generaciones + 1):
            poblacion = generar_nueva_poblacion(poblacion, fitness, porcNewInd, 
                                               probMutacion, funSeleccion, probCruza=probCruza)
            fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
            mejor_fitness_actual = np.min(fitness)
            if mejor_fitness_actual < mejor_fitness_historico:
                mejor_fitness_historico = mejor_fitness_actual
                mejor_individuo = poblacion[np.argmin(fitness)].copy()
            if gen % 10 == 0:
                print(f"Generación {gen}: Mejor fitness = {mejor_fitness_historico:.6f}")
        sol = decodifica_array(mejor_individuo, dim_x, n_bits, a, b)
        fit = mejor_fitness_historico
        ind = mejor_individuo
    else:
        # Usamos cruza de un punto (modifica la función si quieres más variantes)
        from genetico import cruza_un_punto, mutar_flip, generar_poblacion_inicial, evaluar_poblacion, seleccion_ruleta, seleccion_torneo, seleccion_elitismo
        from codificacion import decodifica_array
        funcion, a, b = FUNCIONES[nombre_funcion]
        poblacion = generar_poblacion_inicial(NIND, dim_x, n_bits)
        fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
        mejor_fitness_historico = np.min(fitness)
        mejor_individuo = poblacion[np.argmin(fitness)].copy()
        print(f"Optimizando {nombre_funcion} con dimensión {dim_x}")
        print(f"Generación 0: Mejor fitness = {mejor_fitness_historico:.6f}")
        for gen in range(1, max_generaciones + 1):
            # Reemplazo de peores
            n_new = int(porcNewInd * NIND)
            elite_indices = np.argsort(fitness)[:NIND - n_new]
            elite = poblacion[elite_indices]
            if funSeleccion == 'ruleta':
                padres = seleccion_ruleta(poblacion, fitness, n_new)
            elif funSeleccion == 'torneo':
                padres = seleccion_torneo(poblacion, fitness, n_new)
            elif funSeleccion == 'elitismo':
                padres = seleccion_elitismo(poblacion, fitness, n_new)
            else:
                raise ValueError("Método de selección no reconocido")
            hijos = []
            for i in range(0, n_new, 2):
                p1 = padres[i % len(padres)]
                p2 = padres[(i+1) % len(padres)]
                if np.random.rand() < probCruza:
                    h1, h2 = cruza_un_punto(p1, p2)
                else:
                    h1, h2 = p1.copy(), p2.copy()
                h1 = mutar_flip(h1, prob_mutacion=probMutacion)
                h2 = mutar_flip(h2, prob_mutacion=probMutacion)
                hijos.extend([h1, h2])
            hijos = hijos[:n_new]
            poblacion = np.vstack((elite, hijos))
            fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
            mejor_fitness_actual = np.min(fitness)
            if mejor_fitness_actual < mejor_fitness_historico:
                mejor_fitness_historico = mejor_fitness_actual
                mejor_individuo = poblacion[np.argmin(fitness)].copy()
            if gen % 10 == 0:
                print(f"Generación {gen}: Mejor fitness = {mejor_fitness_historico:.6f}")
        sol = decodifica_array(mejor_individuo, dim_x, n_bits, a, b)
        fit = mejor_fitness_historico
        ind = mejor_individuo
    print("\n--- Resultado Final ---")
    print(f"Mejor fitness: {fit:.6f}")
    print(f"Mejor solución: {sol}")

if __name__ == "__main__":
    main()
