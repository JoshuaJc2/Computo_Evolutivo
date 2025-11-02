import numpy as np
import random
import math
import time
from sudoku import Sudoku, SudokuSolution

def enfriamiento_geometrico(temperatura, alpha):
    return alpha * temperatura

def recocido_simulado(problema, temp_inicial=100.0, alpha=0.9, max_iteraciones = 1000,
                      markov_length=None, max_estancamiento = 650,
                      initial_reheat_factor=3.0, min_reheat_factor=1.5,
                      seed: int = None, max_evaluaciones: int = None):
    # Inicialización
    # reproducibilidad
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        # Generar semilla aleatoria
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Semilla aleatoria generada: {seed}")

    solucion_actual = SudokuSolution(problema)
    fitness_actual = solucion_actual.evaluate()

    mejor_solucion = solucion_actual.copy()
    mejor_fitness = fitness_actual

    temperatura = temp_inicial
    iteracion = 0

     # Calcular Markov Chain Length automáticamente si no se especifica
    if markov_length is None:
        # Contar celdas vacías
        num_vacias = sum(1 for i in range(problema.size)
                       for j in range(problema.size)
                       if not problema.fixed_cells[i, j])
        # L = 2 × número de celdas vacías (mínimo 50)
        markov_length = max(num_vacias * 2, 50)
        print(f"Markov Chain Length calculado: {markov_length} iteraciones por temperatura")

    it_sin_mejora = 0
    #max_temp = temp_inicial * 0.5
    num_recalentar = 0
    total_it_markov = 0
    start_time = time.time()

    print(f"Iniciando SA con Markov Chain y Reheating Decreciente:")
    print(f"  T_inicial: {temp_inicial:.2f}, Alpha: {alpha}")
    print(f"  Markov Length: {markov_length}")
    print(f"  Reheat inicial: {initial_reheat_factor}x, Reheat final: {min_reheat_factor}x")

    # Ciclo principal
    while temperatura > 1e-4 and mejor_fitness > 0 and iteracion < max_iteraciones:
        for it in range(markov_length):
            total_it_markov += 1

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
            razon_temp = temperatura / temp_inicial
            factor_recalentar_actual = min_reheat_factor + (initial_reheat_factor - min_reheat_factor) * razon_temp
            max_recalentar = temp_inicial * 0.5 * razon_temp + temperatura * min_reheat_factor * (1 - razon_temp)

            nueva_temp = temperatura * factor_recalentar_actual
            temperatura = min(nueva_temp, max_recalentar)

            it_sin_mejora = 0
            num_recalentar += 1
            print(f"  Reheat #{num_recalentar} [iter {iteracion}]: "
                  f"T: {temperatura:.2f} (factor: {factor_recalentar_actual:.2f}x, "
                  f"max: {max_recalentar:.2f})")
        else:
            temperatura = enfriamiento_geometrico(temperatura, alpha)

        iteracion += 1
        # Mostrar progreso cada 100 iteraciones externas
        if iteracion % 100 == 0:
            print(f"  Iter {iteracion}: T={temperatura:.3f}, "
                  f"Best={mejor_fitness:.1f}, Current={fitness_actual:.1f}, "
                  f"Evals={total_it_markov:,}")

        # Checar límite de evaluaciones (criterio de término opcional)
        if max_evaluaciones is not None and total_it_markov >= max_evaluaciones:
            print(f"Tope de evaluaciones alcanzado: {total_it_markov}")
            break

    elapsed = time.time() - start_time
    estadisticas = {
        'evaluaciones_totales': total_it_markov,
        'generaciones': iteracion,
        'time': elapsed,
        'seed': seed
    }

    return mejor_solucion, mejor_fitness, estadisticas

# Ejemplo de uso
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ejecutar SA para un ejemplar de sudoku')
    parser.add_argument('--ejemplar', type=str, default=None,
                        help='Ruta al archivo del ejemplar (si no se indica, se usará un grid de ejemplo)')
    parser.add_argument('--seed', type=int, default=None, help='Semilla RNG (opcional)')
    parser.add_argument('--max_iter', type=int, default=1000, help='Número máximo de iteraciones externas')
    parser.add_argument('--max_evals', type=int, default=None, help='Tope de evaluaciones (opcional)')
    args = parser.parse_args()

    if args.ejemplar:
        sudoku = Sudoku.from_file(args.ejemplar)
    else:
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
        sudoku = Sudoku(grid)

    mejor, fitness, stats = recocido_simulado(sudoku,
                                             temp_inicial=100.0,
                                             alpha=0.9,
                                             max_iteraciones=args.max_iter,
                                             max_estancamiento=650,
                                             initial_reheat_factor=3.0,
                                             min_reheat_factor=1.5,
                                             seed=args.seed,
                                             max_evaluaciones=args.max_evals)

    print(f"\nResultado final:")
    print(f"Fitness: {fitness}")
    print(f"Evaluaciones totales: {stats.get('evaluaciones_totales')}")
    print(f"Generaciones ejecutadas: {stats.get('generaciones')}")
    print(f"\nTablero final:\n{mejor.get_grid()}")