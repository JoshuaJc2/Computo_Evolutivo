# README

**Equipo:** TheKingdomPinguin
**Integrantes:** 

 - Castillo Hernández Antonio - 320017438
 - Luna Campos Emiliano - 320292084
 - Juárez Cruz Joshua - 320124516

## Descripción
Se trata de una implementación del algoritmo de Recocido Simulado (Simulated Annealing) para resolver Sudokus, con tres métodos de enfriamiento: geometric (g), slow (s) y linear (l).

## Estructura de `src/`

- `sudoku.py` : Código principal.  

- `run_all.py` : Script que ejecuta todos los ejemplares con cada método de enfriamiento con 10 repeticiones cada uno (dicho num de rep se puede ajustar en el script modificando el num de la variable `repeticiones`).

- `Ejemplares/` : Tableros de prueba (`David_Filmer1.txt`, `Easy1.txt`, `Hard1.txt`, `Medium1.txt`, `SD2.txt`).

## Requisitos
- Python 3.x  
- numpy

## Uso

### Resolver un Sudoku específico
```bash
cd src
python3 sudoku.py Ejemplares/Easy1.txt s
```

Es decir, sigue la estructura de: 

```bash
python3 sudoku.py ruta al archivo.txt metodo_enfriamiento
```


Ahora bien, si se deseea ejecutar todos los metodos de enfriamiento con todos los ejemplares cierto num de repeticiones `n` podemos usar el script de `run_all.py`, donde lo podemos correr de la siguiente manera.

```bash
python3 run_all.py 
```

Estando en la misma carpeta que el archivo `run_all.py`