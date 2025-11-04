"""
graficas_sudoku.py - Genera gráficas para análisis de SA, ILS y Memético en Sudoku
Adaptado del estilo de graficas.py para algoritmos genéticos
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
from scipy.stats import entropy


def cargar_datos():
    """Carga datos de experimentos_sudoku.py"""
    try:
        with open('../output/datos_experimentacion.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("ERROR: Primero ejecuta 'python experimentacion.py'")
        sys.exit(1)
def cargar_datos(path='../output/datos_experimentacion.pkl'):
    """Carga datos desde un pickle dado (ruta por defecto: ../output/datos_experimentacion.pkl)"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de datos: {path}")
        raise


def distancia_hamming(sol1, sol2):
    """Distancia de Hamming entre dos soluciones"""
    return np.sum(np.array(sol1.values) != np.array(sol2.values))


def distancia_euclidiana(sol1, sol2):
    """Distancia Euclidiana entre dos soluciones"""
    return np.sqrt(np.sum((np.array(sol1.values) - np.array(sol2.values)) ** 2))


def calcular_diversidad_hamming(soluciones):
    """Calcula diversidad promedio usando distancia de Hamming"""
    if len(soluciones) < 2:
        return 0.0
    
    distancias = []
    for i in range(len(soluciones)):
        for j in range(i + 1, len(soluciones)):
            dist = distancia_hamming(soluciones[i], soluciones[j])
            distancias.append(dist)
    
    return np.mean(distancias)


def calcular_diversidad_euclidiana(soluciones):
    """Calcula diversidad promedio usando distancia Euclidiana"""
    if len(soluciones) < 2:
        return 0.0
    
    distancias = []
    for i in range(len(soluciones)):
        for j in range(i + 1, len(soluciones)):
            dist = distancia_euclidiana(soluciones[i], soluciones[j])
            distancias.append(dist)
    
    return np.mean(distancias)


def calcular_entropia_hamming(solucion1, solucion2):
    """Calcula la entropía basada en las diferencias de Hamming"""
    if len(solucion1.values) != len(solucion2.values):
        return 0.0
    
    diferencias = [1 if solucion1.values[i] != solucion2.values[i] else 0 
                   for i in range(len(solucion1.values))]
    
    unique, counts = np.unique(diferencias, return_counts=True)
    probabilities = counts / len(diferencias)
    
    return entropy(probabilities, base=2)


def calcular_entropia_poblacion(soluciones):
    """Calcula la entropía promedio de una población de soluciones"""
    if len(soluciones) < 2:
        return 0.0
    
    entropias = []
    for i in range(len(soluciones)):
        for j in range(i + 1, len(soluciones)):
            ent = calcular_entropia_hamming(soluciones[i], soluciones[j])
            entropias.append(ent)
    
    return np.mean(entropias) if entropias else 0.0


# ============================================================================
# GRÁFICA 1: Evolución de aptitud - TODOS los algoritmos juntos
# ============================================================================

def graficar_evolucion_aptitud_todos(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """
    GRÁFICA 1: Evolución de aptitud de los 3 algoritmos en una sola gráfica
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colores = ['#E63946', '#F77F00', '#06A77D']
    markers = ['o', 's', '^']
    
    # SA
    fitness_sa = [e['fitness'] for e in resultados_sa['ejecuciones']]
    ejecuciones = range(1, len(fitness_sa) + 1)
    ax.plot(ejecuciones, fitness_sa, 
            label='Recocido Simulado',
            linewidth=2.5, color=colores[0], marker=markers[0], markersize=4, markevery=3, alpha=0.85)
    
    # ILS
    fitness_ils = [e['fitness'] for e in resultados_ils['ejecuciones']]
    ax.plot(ejecuciones, fitness_ils,
            label='Búsqueda Local Iterada',
            linewidth=2.5, color=colores[1], marker=markers[1], markersize=4, markevery=3, alpha=0.85)
    
    # Memético
    fitness_mem = [e['fitness'] for e in resultados_mem['ejecuciones']]
    ax.plot(ejecuciones, fitness_mem,
            label='Algoritmo Memético',
            linewidth=2.5, color=colores[2], marker=markers[2], markersize=4, markevery=3, alpha=0.85)
    
    ax.set_xlabel('Número de Ejecución', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fitness Final', fontsize=13, fontweight='bold')
    ax.set_title('Evolución de Aptitud - Todos los Algoritmos', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_evolucion_aptitud_todos.png'), dpi=300)
    plt.close()
    print("   1_evolucion_aptitud_todos.png")


# ============================================================================
# GRÁFICA 2: Diversidad (Hamming y Euclidiana)
# ============================================================================

def graficar_diversidad_todos(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """
    GRÁFICA 2: Evolución de diversidad de los 3 algoritmos
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Calcular diversidades
    def calcular_diversidades(resultados):
        soluciones = [e['solucion'] for e in resultados['ejecuciones']]
        div_hamming = calcular_diversidad_hamming(soluciones)
        div_euclidiana = calcular_diversidad_euclidiana(soluciones)
        return div_hamming, div_euclidiana
    
    div_sa_h, div_sa_e = calcular_diversidades(resultados_sa)
    div_ils_h, div_ils_e = calcular_diversidades(resultados_ils)
    div_mem_h, div_mem_e = calcular_diversidades(resultados_mem)
    
    # Subplot 1: Distancia de Hamming
    algoritmos = ['SA', 'ILS', 'Memético']
    diversidades_hamming = [div_sa_h, div_ils_h, div_mem_h]
    colores = ['#E63946', '#F77F00', '#06A77D']
    
    bars1 = ax1.bar(algoritmos, diversidades_hamming, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Diversidad (Hamming)', fontsize=12, fontweight='bold')
    ax1.set_title('Diversidad - Distancia de Hamming', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, diversidades_hamming):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Subplot 2: Distancia Euclidiana
    diversidades_euclidiana = [div_sa_e, div_ils_e, div_mem_e]
    
    bars2 = ax2.bar(algoritmos, diversidades_euclidiana, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Diversidad (Euclidiana)', fontsize=12, fontweight='bold')
    ax2.set_title('Diversidad - Distancia Euclidiana', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, diversidades_euclidiana):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Análisis de Diversidad - Dos Medidas de Distancia',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_diversidad_todos.png'), dpi=300)
    plt.close()
    print("   2_diversidad_todos.png")


# ============================================================================
# GRÁFICA 3: Aptitud y Diversidad combinadas (3 subplots)
# ============================================================================

def graficar_aptitud_diversidad_todos(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """
    GRÁFICA 3: Aptitud y Diversidad combinadas (3 subplots, uno por algoritmo)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    colores_apt = ['#E63946', '#F77F00', '#06A77D']
    colores_div = ['#7B1FA2', '#8E24AA', '#9C27B0']
    
    datos = [
        ('Recocido Simulado', resultados_sa),
        ('Búsqueda Local Iterada', resultados_ils),
        ('Algoritmo Memético', resultados_mem)
    ]
    
    for idx, (nombre, resultado) in enumerate(datos):
        ax1 = axes[idx]
        
        # Calcular diversidad
        soluciones = [e['solucion'] for e in resultado['ejecuciones']]
        diversidad_hamming = []
        for i in range(len(soluciones)):
            div = 0
            count = 0
            for j in range(len(soluciones)):
                if i != j:
                    div += distancia_hamming(soluciones[i], soluciones[j])
                    count += 1
            diversidad_hamming.append(div / count if count > 0 else 0)
        
        fitness_list = [e['fitness'] for e in resultado['ejecuciones']]
        ejecuciones = range(1, len(fitness_list) + 1)
        
        # Eje izquierdo: Fitness
        color1 = colores_apt[idx]
        ax1.set_xlabel('Número de Ejecución', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Fitness', fontsize=10, fontweight='bold', color=color1)
        ax1.plot(ejecuciones, fitness_list, linewidth=2.5, color=color1, marker='o', markersize=3, markevery=3)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Eje derecho: Diversidad
        ax2 = ax1.twinx()
        color2 = colores_div[idx]
        ax2.set_ylabel('Diversidad (Hamming)', fontsize=10, fontweight='bold', color=color2)
        ax2.plot(ejecuciones, diversidad_hamming, linewidth=2.5, color=color2, marker='s', markersize=3, markevery=3)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_title(nombre, fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle('Evolución de Aptitud y Diversidad por Algoritmo', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_aptitud_diversidad_todos.png'), dpi=300)
    plt.close()
    print("   3_aptitud_diversidad_todos.png")


# ============================================================================
# GRÁFICA 4: Calidad de ejecuciones (3 subplots)
# ============================================================================

def graficar_calidad_ejecuciones_todos(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """
    GRÁFICA 4: Calidad de ejecuciones (3 subplots, uno por algoritmo)
    Muestra todas las ejecuciones individuales
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    colores_base = ['#E63946', '#F77F00', '#06A77D']
    
    datos = [
        ('Recocido Simulado', resultados_sa),
        ('Búsqueda Local Iterada', resultados_ils),
        ('Algoritmo Memético', resultados_mem)
    ]
    
    for idx, (nombre, resultado) in enumerate(datos):
        ax = axes[idx]
        
        # Graficar cada ejecución individual
        for ejec in resultado['ejecuciones']:
            ax.scatter(ejec['ejecucion'], ejec['fitness'], 
                      alpha=0.6, color=colores_base[idx], s=100)
        
        # Promedio en negro
        fitness_list = [e['fitness'] for e in resultado['ejecuciones']]
        promedio = np.mean(fitness_list)
        ax.axhline(y=promedio, color='black', linewidth=3, 
                  linestyle='--', label='Promedio')
        
        ax.set_xlabel('Número de Ejecución', fontsize=10, fontweight='bold')
        ax.set_ylabel('Fitness Final', fontsize=10, fontweight='bold')
        ax.set_title(nombre, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
        
        # Estadísticas en texto
        mejor = resultado['mejor']
        peor = resultado['peor']
        prom = resultado['promedio']
        texto = f'Mejor: {mejor:.2f}\nPeor: {peor:.2f}\nProm: {prom:.2f}'
        ax.text(0.98, 0.98, texto, transform=ax.transAxes, fontsize=8,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Calidad de Ejecuciones por Algoritmo', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_calidad_ejecuciones_todos.png'), dpi=300)
    plt.close()
    print("   4_calidad_ejecuciones_todos.png")


# ============================================================================
# GRÁFICA 5: BoxPlot comparativo
# ============================================================================

def graficar_boxplot_comparacion(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """
    GRÁFICA 5: BoxPlot comparando los 3 algoritmos
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    fitness_sa = [e['fitness'] for e in resultados_sa['ejecuciones']]
    fitness_ils = [e['fitness'] for e in resultados_ils['ejecuciones']]
    fitness_mem = [e['fitness'] for e in resultados_mem['ejecuciones']]
    
    datos = [fitness_sa, fitness_ils, fitness_mem]
    labels = ['Recocido\nSimulado', 'Búsqueda Local\nIterada', 'Algoritmo\nMemético']
    
    bp = ax.boxplot(datos, labels=labels, patch_artist=True, notch=True, showmeans=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=3),
                    meanprops=dict(marker='D', markerfacecolor='green', 
                                  markeredgecolor='darkgreen', markersize=10))
    
    # Colorear cajas
    colores = ['#E63946', '#F77F00', '#06A77D']
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Fitness Final', fontsize=13, fontweight='bold')
    ax.set_title('Comparación de Resultados - BoxPlot', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Leyenda
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', alpha=0.7, label='Rango intercuartil (Q1-Q3)'),
        Line2D([0], [0], color='red', linewidth=3, label='Mediana'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='green', 
               markersize=10, label='Media', markeredgecolor='darkgreen'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Bigotes (1.5×IQR)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    # Añadir estadísticas como texto
    y_pos = ax.get_ylim()[1] * 0.95
    for i, r in enumerate([resultados_sa, resultados_ils, resultados_mem]):
        texto = f"μ={r['promedio']:.2f}\nσ={r['desviacion']:.2f}"
        ax.text(i+1, y_pos, texto, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_boxplot_comparacion.png'), dpi=300)
    plt.close()
    print("   5_boxplot_comparacion.png")


# ============================================================================
# GRÁFICA 6: Entropía
# ============================================================================

def graficar_entropia(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """Gráfica de entropía"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    def calcular_entropia_algoritmo(resultados):
        soluciones = [e['solucion'] for e in resultados['ejecuciones']]
        return calcular_entropia_poblacion(soluciones)
    
    entropia_sa = calcular_entropia_algoritmo(resultados_sa)
    entropia_ils = calcular_entropia_algoritmo(resultados_ils)
    entropia_mem = calcular_entropia_algoritmo(resultados_mem)
    
    algoritmos = ['Recocido\nSimulado', 'Búsqueda Local\nIterada', 'Algoritmo\nMemético']
    entropias = [entropia_sa, entropia_ils, entropia_mem]
    colores = ['#E63946', '#F77F00', '#06A77D']
    
    bars = ax.bar(algoritmos, entropias, color=colores, alpha=0.7,
                  edgecolor='black', linewidth=2.5, width=0.6)
    
    for bar, val in zip(bars, entropias):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Entropía (bits)', fontsize=13, fontweight='bold')
    ax.set_title('Análisis de Entropía - Diversidad de Soluciones',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_entropia.png'), dpi=300)
    plt.close()
    print("   6_entropia.png")


def main():
    """Genera las gráficas consolidadas"""
    print("\n" + "="*70)
    print("  GENERANDO GRÁFICAS CONSOLIDADAS")
    print("="*70 + "\n")
    
    datos = cargar_datos()
    resultados_sa = datos['recocido_simulado']
    resultados_ils = datos['ils']
    resultados_mem = datos['memetico']
    
    print("Generando gráficas...\n")
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    graficar_evolucion_aptitud_todos(resultados_sa, resultados_ils, resultados_mem, output_dir)
    graficar_diversidad_todos(resultados_sa, resultados_ils, resultados_mem, output_dir)
    graficar_aptitud_diversidad_todos(resultados_sa, resultados_ils, resultados_mem, output_dir)
    graficar_calidad_ejecuciones_todos(resultados_sa, resultados_ils, resultados_mem, output_dir)
    graficar_boxplot_comparacion(resultados_sa, resultados_ils, resultados_mem, output_dir)
    graficar_entropia(resultados_sa, resultados_ils, resultados_mem, output_dir)
    
    print("\n" + "="*70)
    print("   GRÁFICAS COMPLETADAS")
    print("="*70)
    print("\nArchivos generados en ../output/:")
    print("  1. 1_evolucion_aptitud_todos.png       - Evolución fitness (3 algoritmos)")
    print("  2. 2_diversidad_todos.png              - Diversidad (2 medidas)")
    print("  3. 3_aptitud_diversidad_todos.png      - Aptitud+Diversidad (3 subplots)")
    print("  4. 4_calidad_ejecuciones_todos.png     - Calidad ejecuciones (3 subplots)")
    print("  5. 5_boxplot_comparacion.png           - BoxPlot comparativo")
    print("  6. 6_entropia.png                      - Entropía")
    print("\nTotal: 6 gráficas consolidadas\n")


if __name__ == "__main__":
    main()