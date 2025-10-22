"""
graficas.py - Genera 5 gráficas consolidadas con todas las funciones
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys


def cargar_datos():
    """Carga datos de experimentos.py"""
    try:
        with open('../output/datos_experimentos.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("ERROR: Primero ejecuta 'python experimentos.py'")
        sys.exit(1)


def calcular_promedio_ejecuciones(ejecuciones):
    """Calcula el promedio de múltiples ejecuciones"""
    mejor_prom = np.mean([e['hist_mejor'] for e in ejecuciones], axis=0)
    promedio_prom = np.mean([e['hist_promedio'] for e in ejecuciones], axis=0)
    peor_prom = np.mean([e['hist_peor'] for e in ejecuciones], axis=0)
    div_prom = np.mean([e['hist_diversidad_norm'] for e in ejecuciones], axis=0)
    
    return mejor_prom, promedio_prom, peor_prom, div_prom


def graficar_evolucion_aptitud_todas(resultados):
    """
    GRÁFICA 1: Evolución de aptitud de las 5 funciones
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colores = ['#E63946', '#F77F00', '#06A77D', '#118AB2', '#073B4C']
    markers = ['o', 's', '^', 'D', 'v']
    
    for stats, color, marker in zip(resultados, colores, markers):
        mejor_prom, _, _, _ = calcular_promedio_ejecuciones(stats['ejecuciones'])
        generaciones = range(len(mejor_prom))
        
        ax.plot(generaciones, mejor_prom, 
                label=stats['funcion'].capitalize(),
                linewidth=2.5, color=color, marker=marker, markersize=4, markevery=10, alpha=0.85)
    
    ax.set_xlabel('Generación', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mejor Fitness (promedio)', fontsize=13, fontweight='bold')
    ax.set_title('Evolución de Aptitud - Todas las Funciones', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')  # Escala logarítmica para ver mejor las diferencias
    
    plt.tight_layout()
    plt.savefig('../output/1_evolucion_aptitud_todas.png', dpi=300)
    plt.close()
    print("   1_evolucion_aptitud_todas.png")


def graficar_mejor_vs_promedio_por_funcion(resultados):
    """
    GRÁFICA EXTRA: Para cada función, mejor vs aptitud promedio por generación.
    Cumple explícitamente con el requisito 2.e.
    """
    for stats in resultados:
        func = stats['funcion']
        mejor_prom, promedio_prom, _, _ = calcular_promedio_ejecuciones(stats['ejecuciones'])
        generaciones = range(len(mejor_prom))

        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, mejor_prom, label='Mejor (promedio)', linewidth=2.5, color='#1565C0')
        plt.plot(generaciones, promedio_prom, label='Aptitud promedio', linewidth=2.5, color='#F57C00')
        plt.xlabel('Generación', fontsize=12, fontweight='bold')
        plt.ylabel('Fitness', fontsize=12, fontweight='bold')
        plt.title(f'Evolución de Aptitud - {func.capitalize()}', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../output/evolucion_{func}_mejor_vs_promedio.png', dpi=300)
        plt.close()
        print(f"   evolucion_{func}_mejor_vs_promedio.png")


def graficar_diversidad_todas(resultados):
    """
    GRÁFICA 2: Evolución de diversidad de las 5 funciones
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colores = ['#E63946', '#F77F00', '#06A77D', '#118AB2', '#073B4C']
    markers = ['o', 's', '^', 'D', 'v']
    
    for stats, color, marker in zip(resultados, colores, markers):
        _, _, _, div_prom = calcular_promedio_ejecuciones(stats['ejecuciones'])
        generaciones = range(len(div_prom))
        
        ax.plot(generaciones, div_prom,
                label=stats['funcion'].capitalize(),
                linewidth=2.5, color=color, marker=marker, markersize=4, markevery=10, alpha=0.85)
    
    ax.set_xlabel('Generación', fontsize=13, fontweight='bold')
    ax.set_ylabel('Diversidad Normalizada (promedio)', fontsize=13, fontweight='bold')
    ax.set_title('Evolución de Diversidad - Todas las Funciones', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('../output/2_evolucion_diversidad_todas.png', dpi=300)
    plt.close()
    print("   2_evolucion_diversidad_todas.png")


def graficar_aptitud_diversidad_todas(resultados):
    """
    GRÁFICA 3: Aptitud y Diversidad combinadas (5 subplots, uno por función)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colores_apt = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5']
    colores_div = ['#7B1FA2', '#8E24AA', '#9C27B0', '#AB47BC', '#BA68C8']
    
    for idx, stats in enumerate(resultados):
        ax1 = axes[idx]
        mejor_prom, _, _, div_prom = calcular_promedio_ejecuciones(stats['ejecuciones'])
        generaciones = range(len(mejor_prom))
        
        # Eje izquierdo: Fitness
        color1 = colores_apt[idx]
        ax1.set_xlabel('Generación', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Mejor Fitness', fontsize=10, fontweight='bold', color=color1)
        ax1.plot(generaciones, mejor_prom, linewidth=2.5, color=color1, marker='o', markersize=2, markevery=15)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Eje derecho: Diversidad
        ax2 = ax1.twinx()
        color2 = colores_div[idx]
        ax2.set_ylabel('Diversidad', fontsize=10, fontweight='bold', color=color2)
        ax2.plot(generaciones, div_prom, linewidth=2.5, color=color2, marker='s', markersize=2, markevery=15)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_title(f'{stats["funcion"].capitalize()}', fontsize=12, fontweight='bold', pad=10)
    
    # Ocultar el último subplot
    axes[5].axis('off')
    
    plt.suptitle('Evolución de Aptitud y Diversidad por Función', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('../output/3_aptitud_diversidad_todas.png', dpi=300)
    plt.close()
    print("   3_aptitud_diversidad_todas.png")


def graficar_calidad_ejecuciones_todas(resultados):
    """
    GRÁFICA 4: Calidad de ejecuciones (5 subplots, uno por función)
    Muestra todas las ejecuciones individuales
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colores_base = ['#E63946', '#F77F00', '#06A77D', '#118AB2', '#073B4C']
    
    for idx, stats in enumerate(resultados):
        ax = axes[idx]
        
        # Graficar cada ejecución individual
        for i, ejec in enumerate(stats['ejecuciones']):
            generaciones = range(len(ejec['hist_mejor']))
            ax.plot(generaciones, ejec['hist_mejor'], 
                    linewidth=1.2, alpha=0.4, color=colores_base[idx])
        
        # Promedio en negro
        mejor_prom, _, _, _ = calcular_promedio_ejecuciones(stats['ejecuciones'])
        ax.plot(generaciones, mejor_prom, linewidth=3, color='black', 
                label='Promedio', linestyle='--')
        
        ax.set_xlabel('Generación', fontsize=10, fontweight='bold')
        ax.set_ylabel('Mejor Fitness', fontsize=10, fontweight='bold')
        ax.set_title(f'{stats["funcion"].capitalize()}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')
        
        # Estadísticas en texto
        mejor = stats['mejor']
        peor = stats['peor']
        prom = stats['promedio']
        texto = f'Mejor: {mejor:.4f}\nPeor: {peor:.4f}\nProm: {prom:.4f}'
        ax.text(0.98, 0.98, texto, transform=ax.transAxes, fontsize=8,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Ocultar el último subplot
    axes[5].axis('off')
    
    plt.suptitle('Calidad de Ejecuciones por Función', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('../output/4_calidad_ejecuciones_todas.png', dpi=300)
    plt.close()
    print("   4_calidad_ejecuciones_todas.png")


def graficar_boxplot_comparacion(resultados):
    """
    GRÁFICA 5: BoxPlot comparando las 5 funciones
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    datos = [r['fitness_list'] for r in resultados]
    labels = [r['funcion'].capitalize() for r in resultados]
    
    bp = ax.boxplot(datos, labels=labels, patch_artist=True, notch=True, showmeans=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=3),
                    meanprops=dict(marker='D', markerfacecolor='green', 
                                  markeredgecolor='darkgreen', markersize=10))
    
    # Colorear cajas
    colores = ['#E63946', '#F77F00', '#06A77D', '#118AB2', '#073B4C']
    for patch, color in zip(bp['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Función', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mejor Fitness Final', fontsize=13, fontweight='bold')
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
    for i, r in enumerate(resultados):
        texto = f"μ={r['promedio']:.4f}\nσ={r['desviacion']:.4f}"
        ax.text(i+1, y_pos, texto, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../output/5_boxplot_comparacion.png', dpi=300)
    plt.close()
    print(" 5_boxplot_comparacion.png")


def main():
    """Genera las 5 gráficas consolidadas"""
    print("\n" + "="*70)
    print("  GENERANDO GRÁFICAS CONSOLIDADAS")
    print("="*70 + "\n")
    
    resultados = cargar_datos()
    
    print("Generando gráficas...\n")
    
    graficar_evolucion_aptitud_todas(resultados)
    graficar_diversidad_todas(resultados)
    graficar_aptitud_diversidad_todas(resultados)
    graficar_calidad_ejecuciones_todas(resultados)
    graficar_boxplot_comparacion(resultados)
    graficar_mejor_vs_promedio_por_funcion(resultados)
    
    print("\n" + "="*70)
    print("   GRÁFICAS COMPLETADAS")
    print("="*70)
    print("\nArchivos generados en ../output/:")
    print("  1. 1_evolucion_aptitud_todas.png      - Evolución fitness (5 funciones)")
    print("  2. 2_evolucion_diversidad_todas.png   - Evolución diversidad (5 funciones)")
    print("  3. 3_aptitud_diversidad_todas.png     - Aptitud+Diversidad (5 subplots)")
    print("  4. 4_calidad_ejecuciones_todas.png    - Calidad ejecuciones (5 subplots)")
    print("  5. 5_boxplot_comparacion.png          - BoxPlot comparativo")
    print("  6. evolucion_<func>_mejor_vs_promedio.png - Mejor vs Promedio (1 por función)")
    print("\nTotal: 5 gráficas consolidadas\n")


if __name__ == "__main__":
    main()