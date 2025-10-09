import sys
import numpy as np
import random
from codificacion import decodifica_array
from funciones import sphere, ackley, griewank, rastrigin, rosenbrock

# Seleccion por ruleta
def seleccion(poblacion, fitness_array, num_padres):
    inv_fitness = np.max(fitness_array) - fitness_array + 1
    suma_fitness = np.sum(inv_fitness)

    probas = inv_fitness / suma_fitness
    #probas_acumulativas = np.cumsum(probas)

    #Indices elegidos.
    indices = np.random.choice(len(poblacion), size=num_padres, p=probas)
    return poblacion[indices]

# Cruza uniforme
def cruza(padre1, padre2, prob_cruza=0.8):
    longitud = len(padre1)

    r = random.random()
    if r < prob_cruza:
        hijo1 = np.zeros(longitud, dtype=padre1_array.dtype)
        hijo2 = np.zeros(longitud, dtype=padre2_array.dtype)

        for i in range(longitud):
            bit = random.randint(0, 1)

            if bit == 0:
                hijo1[i] = padre1_array[i]
                hijo2[i] = padre2_array[i]
            else:
                hijo1[i] = padre2_array[i]
                hijo2[i] = padre1_array[i]

    else:
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()

    return hijo1, hijo2

# Mutación
# Implementa una mutacion por intercambio.
def mutar(individuo):
    # Selecciona aleatoriamente dos indices del individuo
    a, b = random.sample(range(len(individuo)), 2)
    # Intercambia sus posiciones
    individuo[a], individuo[b] = individuo[b], individuo[a]
    return individuo

# Funcion que permite generar un nueva poblacion de soluciones para
# el problema de las N reinas.
# poblacion     -   Matriz de tamaño NIND x N, donde NIND es el numero
#                   de individuos por poblacion. Cada entrada identifica un
#                   individuo de la poblacion en la representacion de vectores.
# fitness       -   Matriz con el valor fitness de cada individuos en la poblacion.
# porcNewInd    -   Porcentaje de nuevos individuos a crear en esta generacion.
# porcMutacion  -   Porcentaje de probabilidad con la que se mutara un individuo.
# funSeleccion	-   El método de selección a usar para escoger padres.
def generar_nueva_poblacion(poblacion, fitness, porcNewInd, porcMutacion, funSeleccion):
    NIND = len(poblacion)
    nReinas = poblacion.shape[1]
    n_new = int(porcNewInd * NIND)      # Numero de nuevos individuo para esta poblacion

    # El resto se eligen por elitismo
    elite_indices = np.argsort(fitness)[:NIND - n_new]
    elite = poblacion[elite_indices]

    if funSeleccion.lower() == "ruleta":
        padres = seleccion_ruleta(poblacion, fitness, n_new)
    elif funSeleccion.lower() == "torneo":
        padres = seleccion_torneo(poblacion, fitness, n_new)
    else:
        raise ValueError("Método de selección no reconocido")


    # Generar nuevos individuos por cruce
    hijos = []
    for i in range(0, n_new, 2):
        # Tomar a los padres para el cruce
        p1 = padres[i % len(padres)]
        p2 = padres[(i+1) % len(padres)]
        # Generar un par de hijos
        h1 = cruzar_ox(p1, p2)
        h2 = cruzar_ox(p2, p1)
        # Aleatoriamente decidir si introducir mutacion a algunos de los hijos
        if random.random() < porcMutacion:
            h1 = mutar(h1)
        if random.random() < porcMutacion:
            h2 = mutar(h2)
        # Agregar el nuevo par de hijos
        hijos.extend([h1, h2])
    hijos[:n_new]
    nueva_poblacion = np.vstack((elite, hijos))
    return nueva_poblacion

# Permite generar una poblacion inicial para el AG.
# NIND  -   Numero de individuos en la poblacion.
# N     -   Numero de reinas
# Genera un matriz de vectores de N entradas, cada uno representando un
# tablero de tamaño N x N.
def generar_poblacion_inicial(NIND, N):
    # Genera una matrix NIND x N con entradas 0.
    # poblacion[i] es un vector del modelo
    poblacion = np.zeros((NIND, N), dtype=int)
    for i in range(NIND):
        # Genera individuos aleatorios que son candidatos a solucion
        poblacion[i] = np.random.permutation(N) + 1  # columnas de 1 a N
    return poblacion

