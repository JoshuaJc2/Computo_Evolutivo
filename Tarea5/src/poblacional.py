import random
import time
import numpy as np
from sudoku import Sudoku, SudokuSolution

class GeneticSudoku:
    def __init__(self, problem, pop_size=200, p_cruce=0.8, p_mut=0.03, 
                 torneo_k=3, max_generaciones=500):
        self.problem = problem
        self.pop_size = pop_size
        self.p_cruce = p_cruce
        self.p_mut = p_mut
        self.torneo_k = torneo_k
        self.max_generaciones = max_generaciones

    # --- Inicialización ---
    def inicializar_poblacion(self):
        return [SudokuSolution(self.problem) for _ in range(self.pop_size)]

    # --- Evaluación ---
    def evaluar_poblacion(self, poblacion):
        """Devuelve fitnesses y cuenta evaluaciones"""
        fitnesses = np.zeros(len(poblacion))
        for i, ind in enumerate(poblacion):
            fitnesses[i] = ind.evaluate()
        return fitnesses, len(poblacion)

    # --- Selección por torneo ---
    def seleccion_torneo(self, poblacion, fitnesses):
        indices = random.sample(range(len(poblacion)), self.torneo_k)
        mejor = min(indices, key=lambda i: fitnesses[i])
        return poblacion[mejor]

    # --- Cruza uniforme ---
    def cruza_uniforme(self, p1, p2, prob=0.8):
        if random.random() > prob:
            return SudokuSolution(self.problem, p1.values.copy())
        vals = [
            p1.values[i] if random.random() < 0.5 else p2.values[i]
            for i in range(p1.num_empty)
        ]
        return SudokuSolution(self.problem, vals)

    # --- Mutación por intercambio ---
    def mutacion_intercambio(self, ind, prob=0.01):
        if random.random() > prob or ind.num_empty < 2:
            return
        i, j = random.sample(range(ind.num_empty), 2)
        ind.values[i], ind.values[j] = ind.values[j], ind.values[i]

    # --- Reemplazo elitista ---
    def reemplazo_elitista(self, poblacion, fitnesses, nueva_poblacion, nuevos_fitnesses):
        mejor_idx = np.argmin(fitnesses)
        peor_idx = np.argmax(nuevos_fitnesses)
        nueva_poblacion[peor_idx] = poblacion[mejor_idx].copy()
        nuevos_fitnesses[peor_idx] = fitnesses[mejor_idx]
        return nueva_poblacion, nuevos_fitnesses

    # --- Bucle principal ---
    def run(self, verbose=True):
        poblacion = self.inicializar_poblacion()
        fitnesses, evals = self.evaluar_poblacion(poblacion)
        total_evaluaciones = evals

        mejor = min(poblacion, key=lambda s: s.evaluate())
        mejor_fit = mejor.evaluate()
        if verbose:
            print(f"Búsqueda inicial: Fitness = {mejor_fit:.2f}")

        historia_fitness = [mejor_fit]

        for gen in range(1, self.max_generaciones + 1):
            nueva = []
            while len(nueva) < self.pop_size:
                p1 = self.seleccion_torneo(poblacion, fitnesses)
                p2 = self.seleccion_torneo(poblacion, fitnesses)
                hijo = self.cruza_uniforme(p1, p2, self.p_cruce)
                self.mutacion_intercambio(hijo, self.p_mut)
                nueva.append(hijo)

            nuevos_fitnesses, evals = self.evaluar_poblacion(nueva)
            total_evaluaciones += evals
            nueva, nuevos_fitnesses = self.reemplazo_elitista(
                poblacion, fitnesses, nueva, nuevos_fitnesses
            )

            cand = min(nueva, key=lambda s: s.evaluate())
            cand_fit = cand.evaluate()
            if cand_fit < mejor_fit:
                mejor, mejor_fit = cand, cand_fit

            poblacion, fitnesses = nueva, nuevos_fitnesses
            historia_fitness.append(mejor_fit)

            if verbose and gen % 10 == 0:
                print(f"Generación {gen}: Mejor fitness = {mejor_fit:.2f}")

            if mejor_fit == 0:
                print(f"\n¡Solución óptima encontrada en generación {gen}!")
                break

        estadisticas = {
            'evaluaciones_totales': total_evaluaciones,
            'historia_fitness': historia_fitness,
            'generaciones': gen
        }

        return mejor, mejor_fit, estadisticas


# Ejemplo de uso
if __name__ == "__main__":
    from sudoku import Sudoku

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
    ga = GeneticSudoku(sudoku, pop_size=100, p_cruce=0.8, p_mut=0.01, max_generaciones=100)
    mejor, fitness, stats = ga.run()

    print(f"\nResultado final:")
    print(f"Fitness: {fitness}")
    print(f"Evaluaciones totales: {stats['evaluaciones_totales']}")
    print(f"Generaciones ejecutadas: {stats['generaciones']}")
    print(f"\nTablero final:\n{mejor.get_grid()}")
