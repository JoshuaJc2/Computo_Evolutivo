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
# GRÁFICA EXTRA: Evolución de aptitud - Memético
# ============================================================================

def graficar_evolucion_memetico(resultados_mem, output_dir):
    """
    Gráfica especializada para el memético: muestra la evolución por generación
    de la aptitud de la mejor solución y de la aptitud promedio de la población
    para cada ejecución (líneas sobrepuestas).
    """
    ejecuciones = resultados_mem['ejecuciones']
    if not ejecuciones:
        print("No hay ejecuciones para memético")
        return

    # Preparar figura con dos subplots (mejor y promedio)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    all_hist_mejor = []
    all_hist_prom = []
    for idx, ejec in enumerate(ejecuciones):
        hist_mejor = ejec.get('historia_mejor')
        hist_prom = ejec.get('historia_promedio')
        if hist_mejor is None and hist_prom is None:
            # intentar leer en 'stats' si está anidado
            stats = ejec.get('stats', {})
            hist_mejor = stats.get('historia_fitness')
            hist_prom = stats.get('historia_promedio')

        if hist_mejor:
            generations = list(range(len(hist_mejor)))
            ax1.plot(generations, hist_mejor, alpha=0.4, label=f'Ejec {ejec.get("ejecucion")}', linewidth=1)
            all_hist_mejor.append(hist_mejor)

        if hist_prom:
            generations_prom = list(range(len(hist_prom)))
            ax2.plot(generations_prom, hist_prom, alpha=0.4, label=f'Ejec {ejec.get("ejecucion")}', linewidth=1)
            all_hist_prom.append(hist_prom)

    # Promedio a través de ejecuciones (trazar si hay varias longitudes las truncamos a la mínima)
    if all_hist_mejor:
        min_len = min(len(h) for h in all_hist_mejor)
        mean_mejor = np.mean([h[:min_len] for h in all_hist_mejor], axis=0)
        ax1.plot(range(min_len), mean_mejor, color='black', linewidth=3, label='Media ejecuciones')

    if all_hist_prom:
        min_len = min(len(h) for h in all_hist_prom)
        mean_prom = np.mean([h[:min_len] for h in all_hist_prom], axis=0)
        ax2.plot(range(min_len), mean_prom, color='black', linewidth=3, label='Media ejecuciones')

    ax1.set_ylabel('Fitness mejor (por generación)', fontsize=12, fontweight='bold')
    ax1.set_title('Evolución - Memético: Mejor por generación', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # fitness menor es mejor (si aplica)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Generación', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fitness promedio (por generación)', fontsize=12, fontweight='bold')
    ax2.set_title('Evolución - Memético: Promedio de población por generación', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'memetico_evolucion_generaciones.png')
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"   {os.path.basename(outpath)}")


def graficar_evolucion_comparativa_iter(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """
    Genera una figura con 3 subplots (uno por algoritmo) mostrando:
      - línea de la aptitud de la solución activa / promedio (color)
      - línea del mejor-so-far por iteración (negro grueso)

    Está pensada para num_ejecuciones=1 (trayectorias individuales), pero
    también promedia si hay varias ejecuciones.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)

    datos = [
        ('Recocido Simulado', resultados_sa, axes[0]),
        ('Búsqueda Local Iterada', resultados_ils, axes[1]),
        ('Algoritmo Memético', resultados_mem, axes[2])
    ]

    colores = {'Recocido Simulado': '#E63946', 'Búsqueda Local Iterada': '#F77F00', 'Algoritmo Memético': '#06A77D'}

    for nombre, resultado, ax in datos:
        ejecuciones = resultado.get('ejecuciones', [])
        if not ejecuciones:
            ax.set_title(f"{nombre} (sin ejecuciones)")
            ax.grid(True, alpha=0.3)
            continue

        all_hist_mejor = []
        all_hist_act = []
        for ejec in ejecuciones:
            # mejor
            hist_mejor = ejec.get('historia_mejor') or ejec.get('historia_fitness')
            # activo/promedio: memetico usa 'historia_promedio'
            if nombre == 'Algoritmo Memético':
                hist_act = ejec.get('historia_promedio')
            else:
                hist_act = ejec.get('historia_actual')

            if hist_mejor is not None:
                all_hist_mejor.append(list(hist_mejor))
            if hist_act is not None:
                all_hist_act.append(list(hist_act))

        if not all_hist_mejor and not all_hist_act:
            ax.set_title(f"{nombre} (no hay historiales por iteración)")
            ax.grid(True, alpha=0.3)
            continue

        # Si hay múltiples ejecuciones, truncar a la mínima longitud y promediar
        if all_hist_mejor:
            min_len_mejor = min(len(h) for h in all_hist_mejor)
            mean_mejor = np.mean([h[:min_len_mejor] for h in all_hist_mejor], axis=0)
            ax.plot(range(min_len_mejor), mean_mejor, color='black', linewidth=3, label='Mejor (media ejecuciones)')

        if all_hist_act:
            min_len_act = min(len(h) for h in all_hist_act)
            mean_act = np.mean([h[:min_len_act] for h in all_hist_act], axis=0)
            ax.plot(range(min_len_act), mean_act, color=colores[nombre], linewidth=2.2, label='Activo / Promedio')

        ax.set_title(f'Evolución por iteración - {nombre}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteración / Generación', fontsize=11)
        ax.set_ylabel('Fitness', fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'evolucion_comparativa_iteraciones.png')
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"   {os.path.basename(outpath)}")


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


def graficar_entropia_evolucion(resultados_sa, resultados_ils, resultados_mem, output_dir):
    """Genera 3 gráficas separadas de la evolución de la entropía (una por algoritmo).

        Para cada algoritmo, intenta primero usar poblaciones por iteración/generación.
        Si no existen, usa trayectorias de soluciones (cohortes entre ejecuciones).
        Archivos de salida:
            - 6_entropia_evolucion_sa.png
            - 6_entropia_evolucion_ils.png
            - 6_entropia_evolucion_memetico.png
    """
    def calcular_entropia_por_iteracion(resultados):
        # Cada ejecución puede contener una lista de poblaciones por generación.
        # Buscamos varias formas comunes de almacenarlo.

        # Helper vectorizado: entropía promedio entre pares a partir de una población
        def _entropia_promedio_poblacion_fast(poblacion):
            try:
                # Convertir a matriz (n_individuos, longitud_gen)
                arr = np.array([
                    np.asarray(getattr(sol, 'values', sol), dtype=np.int32)
                    for sol in poblacion
                ], dtype=np.int32)
            except Exception:
                return 0.0

            if arr.ndim != 2 or arr.shape[0] < 2:
                return 0.0

            n, L = arr.shape
            # Matriz de distancias de Hamming por pares (vectorizado)
            # (n,1,L) != (1,n,L) -> (n,n,L) y luego sumamos en eje de genes
            dif = (arr[:, None, :] != arr[None, :, :])
            d = dif.sum(axis=2)  # (n,n)
            # Usar solo triángulo superior (i<j)
            iu = np.triu_indices(n, k=1)
            d_flat = d[iu].astype(np.float64)
            if d_flat.size == 0:
                return 0.0
            p = d_flat / float(L)
            # Entropía binaria H(p) de forma estable
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            return float(h.mean())

        entropias_por_ejec = []
        for ejec in resultados.get('ejecuciones', []):
            poblaciones = ejec.get('poblaciones')
            if poblaciones is None:
                # intentar en stats anidado
                stats = ejec.get('stats') or {}
                poblaciones = stats.get('poblaciones')

            # a veces la estructura viene como lista de listas de soluciones
            if poblaciones and isinstance(poblaciones, list):
                entropias_ejec = []
                for pobl in poblaciones:
                    # Cálculo vectorizado por generación
                    entropias_ejec.append(_entropia_promedio_poblacion_fast(pobl))
                if entropias_ejec:
                    entropias_por_ejec.append(entropias_ejec)

        # Si no hay poblaciones por ejecución, intentar con trayectorias (cohortes entre ejecuciones)
        if not entropias_por_ejec:
            trayectorias = []
            for ejec in resultados.get('ejecuciones', []):
                tray = ejec.get('trayectoria_soluciones')
                if tray is None:
                    stats = ejec.get('stats') or {}
                    tray = stats.get('trayectoria_soluciones')
                if tray and isinstance(tray, list) and len(tray) > 0:
                    trayectorias.append(tray)

            if not trayectorias:
                return []

            # Longitud común mínima
            min_len = min(len(tr) for tr in trayectorias)
            if min_len < 1:
                return []

            entropia_por_t = []
            for t in range(min_len):
                # Cohorte en iteración t: tomar la solución de cada ejecución en t
                pobl_t = []
                for tr in trayectorias:
                    try:
                        pobl_t.append(np.asarray(tr[t], dtype=np.int32))
                    except Exception:
                        pass
                if len(pobl_t) >= 2:
                    entropia_por_t.append(_entropia_promedio_poblacion_fast(pobl_t))
                else:
                    entropia_por_t.append(0.0)
            return entropia_por_t

        # Alinear longitudes (truncar a la mínima) y promediar por posición
        min_len = min(len(e) for e in entropias_por_ejec)
        aligned = [e[:min_len] for e in entropias_por_ejec]
        promedio = [float(sum(vals)) / len(vals) for vals in zip(*aligned)]
        return promedio

    ent_sa = calcular_entropia_por_iteracion(resultados_sa)
    ent_ils = calcular_entropia_por_iteracion(resultados_ils)
    ent_mem = calcular_entropia_por_iteracion(resultados_mem)

    def _plot_single(series, titulo, color, filename):
        if not series:
            print(f"   (info) No hay datos para {titulo}; se omite {filename}")
            return False
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(range(1, len(series) + 1), series, color=color, linewidth=2.2)
        ax.set_xlabel('Iteraciones / Generaciones', fontsize=13, fontweight='bold')
        ax.set_ylabel('Entropía (bits)', fontsize=13, fontweight='bold')
        ax.set_title(f'Evolución de la Entropía - {titulo}', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        outpath = os.path.join(output_dir, filename)
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"   {filename}")
        return True

    any_plotted = False
    any_plotted |= _plot_single(ent_sa, 'Recocido Simulado', '#E63946', '6_entropia_evolucion_sa.png')
    any_plotted |= _plot_single(ent_ils, 'Búsqueda Local Iterada', '#F77F00', '6_entropia_evolucion_ils.png')
    any_plotted |= _plot_single(ent_mem, 'Algoritmo Memético', '#06A77D', '6_entropia_evolucion_memetico.png')

    if not any_plotted:
        print('   (warn) No se generó ninguna gráfica de evolución de entropía (sin datos en SA/ILS/Memético).')


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
    # Evolución de la entropía (líneas) - requiere poblaciones por iteración en los datos
    graficar_entropia_evolucion(resultados_sa, resultados_ils, resultados_mem, output_dir)
    
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