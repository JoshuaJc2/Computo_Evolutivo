import numpy as np
import random
import math
from sudoku import Sudoku, SudokuSolution

def enfriamiento_geometrico(temperatura, alpha):
    return alpha * temperatura

def recocido_simulado(problema, temp_inicial=100.0, alpha=0.0005, N0_factor = 2, p=1.15, max_iteracion = 250000, max_estancamiento = 500, factor_calor = 3.0):
    # Inicialización
    solucion_actual = SudokuSolution(problema)
    fitness_actual = solucion_actual.evaluate()
    mejor_solucion = solucion_actual.copy()
    mejor_fitness = fitness_actual
    alpha = 0.88        # Alpha customizada para geometric

    N = int(N0_factor * problema.size)
    temperatura = temp_inicial
    iteracion = 0

    it_sin_mejora = 0
    max_temp = temp_inicial * 0.5
    num_recal = 0

    print(f"N {N}")
    print(f"Temperatura inicial: {temperatura}")
    # Ciclo principal
    while temperatura > 1e-4 and best_fitness > 0 and iteration < max_iteration:
        #for _ in range(N):
        # Generar vecino
        vecino = solucion_actual.get_neighbor()
        vecino_fitness = vecino.evaluate()

        delta_fitness = vecino_fitness - fitness_actual
        if delta_fitness <= 0:
            acceptar = True
        else:
            probabilidad = math.exp(-delta_fitness / temperatura)
            acceptar = random.random() < probabilidad

        # Actualizar solución actual
        if acceptar:
            solucion_actual = vecino
            fitness_actual = vecino_fitness

            # Actualizar mejor solución
            if fitness_actual < mejor_fitness:
                mejor_solucion = solucion_actual.copy()
                mejor_fitness = fitness_actual
                it_sin_mejora = 0
            else:
                it_sin_mejora += 1
        else:
            it_sin_mejora += 1

        if it_sin_mejora >= max_estancamiento:
            nueva_temp = temperatura * factor_calor
            temperatura = min(nueva_temp, max_temp)
            it_sin_mejora = 0
            num_recal += 1
        else:
            temperatura = enfriamiento_geometrico(temperatura, alpha)

        iteracion += 1
        #N = int(N * p)
    return mejor_solucion, mejor_fitness


