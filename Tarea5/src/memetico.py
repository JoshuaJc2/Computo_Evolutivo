# src/ga_memetico.py
import random
import time
import numpy as np
from sudoku import Sudoku, SudokuSolution
from ils import BusquedaLocalIteradaSudoku  

class MemeticGeneticSudoku:
    def __init__(self, problem,
                 pop_size=100,
                 p_cruce=0.8,
                 p_mut=0.01,
                 torneo_k=3,
                 max_generaciones=100,
                 local_apply_prob=0.2,    # probabilidad de aplicar búsqueda local a cada hijo
                 local_apply_top=0,       # alternativa: número de mejores hijos a mejorar (si > 0)
                 local_max_iter=50,       # max iter para la búsqueda local aplicada a cada individuo
                 seed=None):
        self.problem = problem
        self.pop_size = pop_size
        self.p_cruce = p_cruce
        self.p_mut = p_mut
        self.torneo_k = torneo_k
        self.max_generaciones = max_generaciones

        # Memetic params
        self.local_apply_prob = local_apply_prob
        self.local_apply_top = local_apply_top
        self.local_max_iter = local_max_iter

        # RNG
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        else:
            # Generar semilla aleatoria
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
            print(f"Semilla aleatoria generada: {seed}")

        # ILS instance for local improvements
        self.ils = BusquedaLocalIteradaSudoku(problem)

    def inicializar_poblacion(self):
        return [SudokuSolution(self.problem) for _ in range(self.pop_size)]

    # Evaluación (conteo evaluaciones)

    def evaluar_poblacion(self, poblacion):
        fitnesses = np.zeros(len(poblacion))
        for i, ind in enumerate(poblacion):
            fitnesses[i] = ind.evaluate()
        return fitnesses, len(poblacion)

    
    # Selección por torneo
    def seleccion_torneo(self, poblacion, fitnesses):
        indices = random.sample(range(len(poblacion)), self.torneo_k)
        mejor = min(indices, key=lambda i: fitnesses[i])
        return poblacion[mejor]

    
    # Cruza uniforme 
    def cruza_uniforme(self, p1, p2, prob=0.8):
        if random.random() > prob:
            # sin cruza: copia padre
            return SudokuSolution(self.problem, p1.values.copy())
        vals = [
            p1.values[i] if random.random() < 0.5 else p2.values[i]
            for i in range(p1.num_empty)
        ]
        return SudokuSolution(self.problem, vals)

    
    # Mutación por intercambio 
    def mutacion_intercambio(self, ind, prob=0.01):
        if random.random() > prob or ind.num_empty < 2:
            return
        i, j = random.sample(range(ind.num_empty), 2)
        ind.values[i], ind.values[j] = ind.values[j], ind.values[i]

    
    # Aplica búsqueda local
    def aplicar_busqueda_local(self, individuo, max_iter_local):
        # busqueda_local espera un SudokuSolution y devuelve (solucion, fitness, evaluaciones)
        improved_sol, improved_fit, evals = self.ils.busqueda_local(individuo, max_iter=max_iter_local)
        return improved_sol, improved_fit, evals

    
    # Reemplazo elitista (μ + λ-like)
    def reemplazo_elitista(self, poblacion, fitnesses, nueva_poblacion, nuevos_fitnesses):
        # Conserva el mejor antiguo (copia sobre el peor nuevo)
        mejor_idx = np.argmin(fitnesses)
        peor_idx = np.argmax(nuevos_fitnesses)
        nueva_poblacion[peor_idx] = poblacion[mejor_idx].copy()
        nuevos_fitnesses[peor_idx] = fitnesses[mejor_idx]
        return nueva_poblacion, nuevos_fitnesses

    # Bucle principal
    def run(self, verbose=True):
        poblacion = self.inicializar_poblacion()
        fitnesses, evals = self.evaluar_poblacion(poblacion)
        total_evaluaciones = evals

        mejor = min(poblacion, key=lambda s: s.evaluate())
        mejor_fit = mejor.evaluate()
        historia_fitness = [mejor_fit]

        if verbose:
            print(f"Búsqueda inicial: Fitness = {mejor_fit:.2f}")

        for gen in range(1, self.max_generaciones + 1):
            nueva = []

            # Generar hijos
            while len(nueva) < self.pop_size:
                p1 = self.seleccion_torneo(poblacion, fitnesses)
                p2 = self.seleccion_torneo(poblacion, fitnesses)
                hijo = self.cruza_uniforme(p1, p2, self.p_cruce)
                self.mutacion_intercambio(hijo, self.p_mut)
                nueva.append(hijo)

            # Posibilidad 1: aplicar búsqueda local a los mejores hijos 
            evals_from_local = 0
            if self.local_apply_top > 0:
                # ordenar hijos por fitness rápido (evaluar primero)
                hijos_fitness, e = self.evaluar_poblacion(nueva)
                total_evaluaciones += e
                evals_from_local += e
                # índices de los mejores hijos
                sorted_idx = np.argsort(hijos_fitness)
                top_k = min(self.local_apply_top, len(nueva))
                for idx in sorted_idx[:top_k]:
                    sol, fit, ev = self.aplicar_busqueda_local(nueva[idx], self.local_max_iter)
                    nueva[idx] = sol
                    hijos_fitness[idx] = fit
                    total_evaluaciones += ev
                    evals_from_local += ev

                nuevos_fitnesses = hijos_fitness

            else:
                # Posibilidad 2: aplicar búsqueda local con probabilidad por hijo
                nuevos_fitnesses = np.zeros(len(nueva))
                # evaluar población de hijos y opcionalmente mejorar algunos
                for i, h in enumerate(nueva):
                    # decidir si aplicar búsqueda local
                    if random.random() < self.local_apply_prob:
                        sol, fit, ev = self.aplicar_busqueda_local(h, self.local_max_iter)
                        nueva[i] = sol
                        nuevos_fitnesses[i] = fit
                        total_evaluaciones += ev
                        evals_from_local += ev
                    else:
                        # evaluamos sin búsqueda local
                        nuevos_fitnesses[i] = h.evaluate()
                        total_evaluaciones += 1

            # Reemplazo elitista
            nueva, nuevos_fitnesses = self.reemplazo_elitista(poblacion, fitnesses, nueva, nuevos_fitnesses)

            # Actualizar mejor
            cand = min(nueva, key=lambda s: s.evaluate())
            cand_fit = cand.evaluate()
            if cand_fit < mejor_fit:
                mejor, mejor_fit = cand, cand_fit

            poblacion, fitnesses = nueva, nuevos_fitnesses
            historia_fitness.append(mejor_fit)

            if verbose and gen % 10 == 0:
                print(f"Generación {gen}: Mejor fitness = {mejor_fit:.2f} (evals totales {total_evaluaciones})")

            if mejor_fit == 0:
                if verbose:
                    print(f"\n¡Solución óptima encontrada en generación {gen}!")
                break

        estadisticas = {
            'evaluaciones_totales': total_evaluaciones,
            'historia_fitness': historia_fitness,
            'generaciones': gen
        }
        return mejor, mejor_fit, estadisticas


# Ejemplo de uso / CLI simple
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ejecutar algoritmo memético para un ejemplar de sudoku')
    parser.add_argument('--ejemplar', type=str, default=None, help='Ruta al ejemplar (archivo)')
    parser.add_argument('--seed', type=int, default=None, help='Semilla RNG (opcional)')
    parser.add_argument('--pop_size', type=int, default=100, help='Tamaño de población')
    parser.add_argument('--max_gen', type=int, default=200, help='Máx generaciones')
    parser.add_argument('--local_apply_prob', type=float, default=0.5, help='Probabilidad de aplicar búsqueda local a cada hijo')
    parser.add_argument('--local_max_iter', type=int, default=100, help='Máx iter. búsqueda local para cada hijo')
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

    ga_mem = MemeticGeneticSudoku(sudoku,
                                  pop_size=args.pop_size,
                                  max_generaciones=args.max_gen,
                                  local_apply_prob=args.local_apply_prob,
                                  local_max_iter=args.local_max_iter,
                                  seed=args.seed)

    mejor, fit, stats = ga_mem.run(verbose=True)

    print("\nResultado final:")
    print(f"La semilla utilizada fue: {args.seed}")
    print(f"Fitness: {fit}")
    print(f"Evaluaciones totales: {stats['evaluaciones_totales']}")
    print(f"Generaciones ejecutadas: {stats['generaciones']}")
    print(f"\nTablero final:\n{mejor.get_grid()}")
