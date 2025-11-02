import numpy as np
import random
from collections import Counter
from sudoku import Sudoku, SudokuSolution

class BusquedaLocalIteradaSudoku:
    def __init__(self, problema_sudoku, seed: int = None):
        """
        Inicializa la búsqueda local iterada para Sudoku
        
        Args:
            problema_sudoku: Instancia de la clase Sudoku
        """
        self.problema = problema_sudoku
        self.seed = seed
        if seed is not None:
            import random as _rand
            import numpy as _np
            _rand.seed(seed)
            _np.random.seed(seed)
        
        else:
            # Generar semilla aleatoria
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
            print(f"Semilla aleatoria generada: {seed}")
        
    def busqueda_local(self, solucion, max_iter=1000):

        mejor_solucion = solucion.copy()
        mejor_fitness = mejor_solucion.evaluate()
        evaluaciones = 1
        
        for iteracion in range(max_iter):
            # Generar vecinos intercambiando pares de valores
            mejora_encontrada = False
            
            # Intentar múltiples intercambios aleatorios
            for _ in range(min(50, solucion.num_empty)):
                vecino = mejor_solucion.get_neighbor()
                fitness_vecino = vecino.evaluate()
                evaluaciones += 1
                
                if fitness_vecino < mejor_fitness:
                    mejor_solucion = vecino
                    mejor_fitness = fitness_vecino
                    mejora_encontrada = True
                    break
            
            # Si no hay mejora, terminar búsqueda local
            if not mejora_encontrada:
                break
        
        return mejor_solucion, mejor_fitness, evaluaciones
    
    def perturbar_debil(self, solucion, intensidad=2):
        perturbada = solucion.copy()
        
        for _ in range(intensidad):
            if perturbada.num_empty >= 2:
                idx1, idx2 = random.sample(range(perturbada.num_empty), 2)
                perturbada.values[idx1], perturbada.values[idx2] = \
                    perturbada.values[idx2], perturbada.values[idx1]
        
        return perturbada
    
    def iterated_local_search(self, max_iter=100, max_iter_local=1000):

        s0 = SudokuSolution(self.problema)
        
        # Línea 2: sactual = búsqueda Local (s0)
        solucion_actual, fitness_actual, evals = self.busqueda_local(s0, max_iter_local)
        
        # Línea 3: smejor = sactual
        mejor_solucion = solucion_actual.copy()
        mejor_fitness = fitness_actual
        
        # Estadísticas
        evaluaciones_totales = evals
        historia_fitness = [mejor_fitness]
        iteraciones_sin_mejora = 0
        
        
        print(f"Búsqueda inicial: Fitness = {fitness_actual:.2f}")
        
        # Línea 4-5: t = 1; mientras Condiciones de término hacer
        t = 1
        while t <= max_iter:

            s_perturbada = self.perturbar_debil(solucion_actual)
            s_hc, fitness_hc, evals = self.busqueda_local(s_perturbada, max_iter_local)
            evaluaciones_totales += evals
            
            # Línea 9-10: si f(shc) < f(smejor) entonces smejor = shc
            if fitness_hc < mejor_fitness:
                mejor_solucion = s_hc.copy()
                mejor_fitness = fitness_hc
                iteraciones_sin_mejora = 0
                print(f"Iteración {t}: *** Nueva mejor solución: {mejor_fitness:.2f} ***")
            else:
                iteraciones_sin_mejora += 1
            

            if fitness_hc < fitness_actual:
                solucion_actual = s_hc
                fitness_actual = fitness_hc
            
            
            historia_fitness.append(mejor_fitness)
            
            # Imprimir progreso
            if t % 10 == 0:
                print(f"Iteración {t}: Mejor = {mejor_fitness:.2f}, Actual = {fitness_actual:.2f}")
            
            # Terminar si encontramos solución óptima
            if mejor_fitness == 0:
                print(f"\n¡Solución óptima encontrada en iteración {t}!")
                break
            
            # Línea 12: t = t + 1
            t += 1
        
        # Línea 13: devolver smejor
        estadisticas = {
            'evaluaciones_totales': evaluaciones_totales,
            'historia_fitness': historia_fitness,
            'iteraciones_totales': t,
            'seed': self.seed
        }
        
        return mejor_solucion, mejor_fitness, estadisticas



# Ejemplo de uso
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ejecutar ILS para un ejemplar de sudoku')
    parser.add_argument('--ejemplar', type=str, default=None, help='Ruta al ejemplar (archivo)')
    parser.add_argument('--seed', type=int, default=None, help='Semilla RNG (opcional)')
    parser.add_argument('--max_iter', type=int, default=100, help='Número máximo de iteraciones ILS')
    parser.add_argument('--max_iter_local', type=int, default=1000, help='It. max para búsqueda local interna')
    args = parser.parse_args()

    if args.ejemplar:
        problema = Sudoku.from_file(args.ejemplar)
    else:
        # grid de ejemplo
        grid_ejemplo = [
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
        problema = Sudoku(grid_ejemplo)

    ils = BusquedaLocalIteradaSudoku(problema, seed=args.seed)
    solucion, fitness, stats = ils.iterated_local_search(max_iter=args.max_iter, max_iter_local=args.max_iter_local)

    print(f"\nResultado final:")
    print(f"Fitness: {fitness}")
    print(f"Evaluaciones totales: {stats['evaluaciones_totales']}")
    print(f"Tablero final:\n{solucion.get_grid()}")