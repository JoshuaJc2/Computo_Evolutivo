import numpy as np
import random
import math
from collections import Counter

class Sudoku:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)       # Copia del tablero inicial
        self.size = self.grid.shape[0]              # n tamaño del sudoku (n x n)

        self.block_size = int(math.sqrt(self.size)) # Tamaño de bloque k = sqrt(n)
        if self.block_size ** 2 != self.size:
            raise ValueError(f"La dimensión {self.size} no es un cuadrado perfecto")

        self.fixed_cells = (self.grid != 0)

    @classmethod
    def from_file(cls, filename):
        file = open(filename,'r')
        lines = [line.strip() for line in file.readlines() if line.strip()]

        grid = []
        for line in lines:
            row = [int(x) for x in line.split()]
            grid.append(row)

        return cls(grid)

    def __str__(self):
        return str(self.grid)

class SudokuSolution:
    def __init__(self, problem, values=None):
        self.problem = problem

        # Obtener posiciones de celdas vacías
        self.empty_positions = []
        for i in range(problem.size):
            for j in range(problem.size):
                if not problem.fixed_cells[i, j]:
                    self.empty_positions.append((i, j))
        self.num_empty = len(self.empty_positions)      # Numero de celdas vacías
        self.position_to_index = {pos: idx for idx, pos in enumerate(self.empty_positions)}

        if values is not None:
            if len(values) != self.num_empty:
                raise ValueError(f"Se esperaban {self.num_empty} valores, se recibieron {len(values)}")
            self.values = list(values)
        else:
            self.values = self._generate_random_solution()

    def _generate_random_solution(self):
        n = self.problem.size

        # Contar valores fijos en el tablero
        fixed_counts = Counter()
        for i in range(n):
            for j in range(n):
                if self.problem.fixed_cells[i, j]:
                    value = self.problem.grid[i, j]
                    fixed_counts[value] += 1

        # Calcular cuántos de cada valor necesitamos agregar
        needed_counts = {}
        for value in range(1, n + 1):
            current_count = fixed_counts.get(value, 0)
            needed = n - current_count
            needed_counts[value] = needed

        # Crear lista de valores necesarios para completar
        values_needed = []
        for value, count in needed_counts.items():
            values_needed.extend([value] * count)

        # Mezclar aleatoriamente los valores necesarios
        random.shuffle(values_needed)

        return values_needed

    def get_value(self, row, col):
        if self.problem.fixed_cells[row, col]:
            return self.problem.grid[row, col]
        else:
            idx = self.position_to_index[(row, col)]
            return self.values[idx]

    def get_row(self, row):
        return [self.get_value(row, col) for col in range(self.problem.size)]

    def get_column(self, col):
        return [self.get_value(row, col) for row in range(self.problem.size)]

    def get_block(self, block_row, block_col):
        values = []
        start_row = block_row * self.problem.block_size
        start_col = block_col * self.problem.block_size

        for i in range(start_row, start_row + self.problem.block_size):
            for j in range(start_col, start_col + self.problem.block_size):
                values.append(self.get_value(i, j))

        return values

    def get_grid(self):
        grid = self.problem.grid.copy()

        for idx, (row, col) in enumerate(self.empty_positions):
            grid[row, col] = self.values[idx]

        return grid

    def evaluate(self):
        n = self.problem.size
        k = self.problem.block_size
        total_conflicts = 0

        # 1. Colisiones en filas
        for row in range(n):
            values = self.get_row(row)
            total_conflicts += self._count_conflicts_in_group(values)

        # 2. Colisiones en columnas
        for col in range(n):
            values = self.get_column(col)
            total_conflicts += self._count_conflicts_in_group(values)

        # 3. Colisiones en bloques
        for block_row in range(k):
            for block_col in range(k):
                values = self.get_block(block_row, block_col)
                total_conflicts += self._count_conflicts_in_group(values)

        return float(total_conflicts)

    def _count_conflicts_in_group(self, values):
        freq = Counter(values)
        return sum(max(0, count - 1) for count in freq.values())

    def copy(self):
        return SudokuSolution(self.problem, self.values.copy())


    def get_neighbor(self):
        if self.num_empty < 2:
            return self.copy()
        # Crear copia de la solución actual
        neighbor = self.copy()
        idx1, idx2 = random.sample(range(self.num_empty), 2)

        neighbor.values[idx1], neighbor.values[idx2] = neighbor.values[idx2], neighbor.values[idx1]
        return neighbor
