
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def cargar_datos(filename='resultados_experimentacion_total.csv'):
    try:
        df = pd.read_csv(filename)
        print(f"✓ Datos cargados: {len(df)} registros")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encontró '{filename}'")
        print("Primero ejecuta el script de experimentación.")
        sys.exit(1)


def crear_boxplot(df_filtrado, variante, F, CR, output_dir='graficas_boxplot'):
    colores = {
        'sphere': '#E63946',
        'ackley': '#F77F00',
        'griewank': '#06A77D',
        'rastrigin': '#118AB2',
        'rosenbrock': '#073B4C'
    }
    
    funciones = df_filtrado['funcion'].unique()
    funciones_ordenadas = ['sphere', 'ackley', 'griewank', 'rastrigin', 'rosenbrock']
    
    datos = []
    labels = []
    colors = []
    
    for func in funciones_ordenadas:
        if func in funciones:
            valores = df_filtrado[df_filtrado['funcion'] == func]['mejor_fitness'].values
            datos.append(valores)
            labels.append(func.capitalize())
            colors.append(colores[func])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bp = ax.boxplot(datos, labels=labels, patch_artist=True, notch=True, showmeans=True,
                    boxprops=dict(linewidth=1.8),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2.5),
                    meanprops=dict(marker='D', markerfacecolor='yellow', 
                                  markeredgecolor='darkorange', markersize=8))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Función', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mejor Fitness (10 repeticiones)', fontsize=13, fontweight='bold')
    
    titulo = f'{variante} - F={F}, CR={CR}'
    ax.set_title(titulo, fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    max_val = max([max(d) for d in datos])
    min_val = min([min(d) for d in datos])
    if max_val / min_val > 100:
        ax.set_yscale('log')
    
    # Leyenda
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', alpha=0.7, label='IQR (Q1-Q3)'),
        Line2D([0], [0], color='red', linewidth=2.5, label='Mediana'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow', 
               markersize=8, label='Media', markeredgecolor='darkorange'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    y_max = ax.get_ylim()[1]
    for i, (func, valores) in enumerate(zip(funciones_ordenadas, datos)):
        media = np.mean(valores)
        std = np.std(valores)
        texto = f'μ={media:.2e}\nσ={std:.2e}'
        
        x_pos = i + 1
        y_pos = y_max * 0.95
        
        ax.text(x_pos, y_pos, texto, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    filename = f'{variante.replace("/", "_")}__F{F}_CR{CR}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ {filename}")


def generar_todas_las_graficas(df):

    output_dir = 'graficas_boxplot'
    os.makedirs(output_dir, exist_ok=True)
    
    variantes = df['variante'].unique()
    configuraciones = df[['F', 'CR']].drop_duplicates().values.tolist()
    
    print(f"\nVariantes encontradas: {list(variantes)}")
    print(f"Configuraciones encontradas: {list(configuraciones)}")
    print(f"\nGenerando {len(variantes)} × {len(configuraciones)} = {len(variantes) * len(configuraciones)} gráficas...\n")
    
    contador = 0
    for variante in sorted(variantes):
        for F, CR in sorted(configuraciones):
            df_filtrado = df[(df['variante'] == variante) & 
                           (df['F'] == F) & 
                           (df['CR'] == CR)]
            
            if len(df_filtrado) > 0:
                crear_boxplot(df_filtrado, variante, F, CR, output_dir)
                contador += 1
    
    return contador, output_dir


def generar_resumen_estadistico(df, output_dir='graficas_boxplot'):

    resumen = []
    
    variantes = df['variante'].unique()
    configuraciones = df[['F', 'CR']].drop_duplicates().values.tolist()
    funciones = df['funcion'].unique()
    
    for variante in sorted(variantes):
        for F, CR in sorted(configuraciones):
            for funcion in sorted(funciones):
                df_filtrado = df[(df['variante'] == variante) & 
                               (df['F'] == F) & 
                               (df['CR'] == CR) &
                               (df['funcion'] == funcion)]
                
                if len(df_filtrado) > 0:
                    valores = df_filtrado['mejor_fitness'].values
                    resumen.append({
                        'variante': variante,
                        'F': F,
                        'CR': CR,
                        'funcion': funcion,
                        'mejor': np.min(valores),
                        'peor': np.max(valores),
                        'media': np.mean(valores),
                        'mediana': np.median(valores),
                        'std': np.std(valores),
                        'q1': np.percentile(valores, 25),
                        'q3': np.percentile(valores, 75)
                    })
    
    df_resumen = pd.DataFrame(resumen)
    resumen_file = os.path.join(output_dir, 'resumen_estadistico.csv')
    df_resumen.to_csv(resumen_file, index=False)
    print(f"\n✓ Resumen estadístico guardado en: {resumen_file}")


def main():
    print("\n" + "="*70)
    print("  GENERADOR DE GRÁFICAS BOXPLOT - EXPERIMENTACIÓN ED")
    print("="*70)
    
    df = cargar_datos('resultados_experimentacion_total.csv')
    
    print(f"\nFunciones: {sorted(df['funcion'].unique())}")
    print(f"Variantes: {sorted(df['variante'].unique())}")
    print(f"Configuraciones (F, CR): {sorted(df[['F', 'CR']].drop_duplicates().values.tolist())}")
    print(f"Repeticiones por configuración: {df.groupby(['variante', 'F', 'CR', 'funcion']).size().iloc[0]}")
    
    num_graficas, output_dir = generar_todas_las_graficas(df)
    
    generar_resumen_estadistico(df, output_dir)
    
    print("\n" + "="*70)
    print(f"  COMPLETADO: {num_graficas} GRÁFICAS GENERADAS")
    print("="*70)
    print(f"\nArchivos guardados en: ./{output_dir}/")
    print("\nFormato de nombres:")
    print("  <VARIANTE>__F<valor>_CR<valor>.png")
    print("\nEjemplos:")
    print("  DE_rand_1__F0.9_CR0.5.png")
    print("  DE_best_1__F0.8_CR0.7.png")
    print("  DE_current-to-best_1__F0.75_CR0.6.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()