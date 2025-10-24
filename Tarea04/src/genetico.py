import sys
import numpy as np
import random
from codificacion import decodifica_array
from funciones import sphere, ackley, griewank, rastrigin, rosenbrock


FUNCIONES = {
    'sphere': (sphere, -5.12, 5.12),
    'ackley': (ackley, -30, 30),
    'griewank': (griewank, -600, 600),
    'rastrigin': (rastrigin, -5.12, 5.12),
    'rosenbrock': (rosenbrock, -2.048, 2.048)
}

def calcular_fitness(individuo, funcion, dim_x, n_bits, a, b):
    x = decodifica_array(individuo, dim_x, n_bits, a, b)
    return funcion(np.array(x))

def evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b):
    fitness_array = np.zeros(len(poblacion))
    for i in range(len(poblacion)):
        fitness_array[i] = calcular_fitness(poblacion[i], funcion, dim_x, n_bits, a, b)
    return fitness_array

# Seleccion por ruleta
def seleccion_ruleta(poblacion, fitness_array, num_padres):
    inv_fitness = np.max(fitness_array) - fitness_array + 1
    suma_fitness = np.sum(inv_fitness)

    probas = inv_fitness / suma_fitness
    #probas_acumulativas = np.cumsum(probas)

    #Indices elegidos.
    indices = np.random.choice(len(poblacion), size=num_padres, p=probas)
    return poblacion[indices]

def seleccion_torneo(population, fitnesses, num_selections, tournament_size=3, pressure=1.0):
    selected = []
    for _ in range(num_selections):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        # Minimización: menor fitness es mejor
        tournament_indices.sort(key=lambda i: fitnesses[i], reverse=False)
        if random.random() < pressure:
            winner = tournament_indices[0]
        else:
            winner = random.choice(tournament_indices[1:] if len(tournament_indices) > 1 else tournament_indices)
        selected.append(population[winner])
    return selected

def seleccion_elitismo(population, fitnesses, num_selections, num_elites=2, tournament_size=3):
    # Minimización: menor fitness es mejor
    sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=False)
    elite_individuals = [population[i] for i in sorted_indices[:num_elites]]
    num_tournament_selections = num_selections - num_elites
    if num_tournament_selections > 0:
        selected_individuals = seleccion_torneo(
            population, 
            fitnesses,
            num_tournament_selections,
            tournament_size
        )
        return elite_individuals + selected_individuals
    else:
        return elite_individuals

# Cruza uniforme
def cruza_uniforme(padre1, padre2, prob_cruza=0.8):
    longitud = len(padre1)
    
    r = random.random()
    if r < prob_cruza:
        hijo1 = np.zeros(longitud, dtype=int)
        hijo2 = np.zeros(longitud, dtype=int)
        
        for i in range(longitud):
            bit = random.randint(0, 1)
            
            if bit == 0:
                hijo1[i] = padre1[i]
                hijo2[i] = padre2[i]
            else:
                hijo1[i] = padre2[i]
                hijo2[i] = padre1[i]
    else:
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()
    
    return hijo1, hijo2


def cruza_un_punto(padre1, padre2):
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
    
    return hijo1, hijo2

# Mutación
# Implementa una mutacion por intercambio.
def mutar(individuo):
    # Selecciona aleatoriamente dos indices del individuo
    a, b = random.sample(range(len(individuo)), 2)
    # Intercambia sus posiciones
    individuo[a], individuo[b] = individuo[b], individuo[a]
    return individuo

def mutar_un_bit(individuo):
    i = random.randrange(len(individuo))
    individuo[i] = 1 - individuo[i]
    return individuo

def mutar_flip(individuo, prob_mutacion=0.01):
    for i in range(len(individuo)):
        if random.random() < prob_mutacion:
            individuo[i] = 1 - individuo[i]
    return individuo


def generar_nueva_poblacion(poblacion, fitness, porcNewInd, porcMutacion, funSeleccion,
                            funCruza, probCruza=0.8):
    NIND = len(poblacion)
    n_new = int(porcNewInd * NIND)
    elite_indices = np.argsort(fitness)[:NIND - n_new]
    elite = poblacion[elite_indices]
    if funSeleccion.lower() == "ruleta":
        padres = seleccion_ruleta(poblacion, fitness, n_new)
    elif funSeleccion.lower() == "torneo":
        padres = seleccion_torneo(poblacion, fitness, n_new)
    elif funSeleccion.lower() == "elitismo":
        padres = seleccion_elitismo(poblacion, fitness, n_new)
    else:
        raise ValueError("Método de selección no reconocido")
    hijos = []
    for i in range(0, n_new, 2):
        p1 = padres[i % len(padres)]
        p2 = padres[(i+1) % len(padres)]
        # Tipo de Cruza
        #h1, h2 = cruza_un_punto(p1, p2)
        if funCruza.lower() == "cruza_un_punto":
            h1, h2 = cruza_un_punto(p1, p2)
        elif funCruza.lower() == "cruza_uniforme":
            h1, h2 = cruza_uniforme(p1, p2, prob_cruza=probCruza)
        h1 = mutar_flip(h1, prob_mutacion=porcMutacion)
        h2 = mutar_flip(h2, prob_mutacion=porcMutacion)
        hijos.extend([h1, h2])
    hijos = hijos[:n_new]
    nueva_poblacion = np.vstack((elite, hijos))
    return nueva_poblacion

def generar_poblacion_inicial(NIND, dim_x, n_bits):
    
    longitud_individuo = dim_x * n_bits
    poblacion = np.random.randint(0, 2, size=(NIND, longitud_individuo))
    return poblacion

def algoritmo_genetico(nombre_funcion, dim_x=10, n_bits=16, NIND=100, 
                       max_generaciones=100, porcNewInd=0.8, probMutacion=0.01,
                       funSeleccion='ruleta', probCruza=0.8, funCruza='cruza_uniforme'):
    # Obtener función y rangos
    funcion, a, b = FUNCIONES[nombre_funcion.lower()]
    
    # Generar población inicial
    poblacion = generar_poblacion_inicial(NIND, dim_x, n_bits)
    
    # Evaluar población inicial
    fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
    
    mejor_fitness_historico = np.min(fitness)
    mejor_individuo = poblacion[np.argmin(fitness)].copy()
    
    print(f"Optimizando {nombre_funcion} con dimensión {dim_x}")
    print(f"Generación 0: Mejor fitness = {mejor_fitness_historico:.6f}")
    print(f"funSeleccion: {funSeleccion}\nfunCruza: {funCruza}")

    
    # Evolución
    for gen in range(1, max_generaciones + 1):
        poblacion = generar_nueva_poblacion(poblacion, fitness, porcNewInd, 
                                           probMutacion, funSeleccion, funCruza,probCruza=probCruza)
        
        # Evaluar nueva población
        fitness = evaluar_poblacion(poblacion, funcion, dim_x, n_bits, a, b)
        
        # Actualizar mejor solución
        mejor_fitness_actual = np.min(fitness)
        if mejor_fitness_actual < mejor_fitness_historico:
            mejor_fitness_historico = mejor_fitness_actual
            mejor_individuo = poblacion[np.argmin(fitness)].copy()
        
        # Mostrar progreso cada 10 generaciones
        if gen % 10 == 0:
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness_historico:.6f}")
    
    # Decodificar mejor solución
    mejor_x = decodifica_array(mejor_individuo, dim_x, n_bits, a, b)
    
    print("\n--- Resultado Final ---")
    print(f"Mejor fitness: {mejor_fitness_historico:.6f}")
    print(f"Mejor solución: {mejor_x}")
    
    return mejor_individuo, mejor_fitness_historico, mejor_x


if __name__ == "__main__":
    # Optimizar función Sphere
    algoritmo_genetico('rosenbrock')
