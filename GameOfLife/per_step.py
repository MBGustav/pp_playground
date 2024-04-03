import sys
import pandas as pd
import matplotlib.pyplot as plt

def criar_grafico(argv):
    if len(argv) != 4:
        print("Uso: python script.py arquivo.csv num_generations size")
        return

    # Parâmetros da linha de comando
    arquivo_csv = argv[1]
    num_generations_fixo = int(argv[2])
    size_fixo = int(argv[3])

    # Leia o arquivo CSV
    df = pd.read_csv(arquivo_csv)

    # Filtre os dados com base no tamanho fixo
    df_filtrado = df[(df['NumberOfGenerations'] == num_generations_fixo) & (df['size'] == size_fixo)]

    # Agrupe os dados por 'type' e 'function' e calcule a soma do tempo para cada grupo
    grouped_data = df_filtrado.groupby(['type', 'function']).agg({'time_ms': 'sum', 'NumberOfGenerations': 'sum'}).reset_index()

    # Multiplique o valor de 'Avg. per Kernel' pelo total de gerações
    grouped_data['time_ms'] = grouped_data.apply(lambda row: row['time_ms'] * row['NumberOfGenerations'] if row['function'] == 'Avg. per Kernel' else row['time_ms'], axis=1)

    # Crie uma tabela pivot para facilitar o plot
    pivot_table = pd.pivot_table(grouped_data, values='time_ms', index='type', columns='function', fill_value=0)

    # Normalize os valores para percentuais
    pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    # Crie o gráfico de barras empilhadas
    ax = pivot_table_percentage.plot(kind='bar', stacked=True, figsize=(10, 6))
    ax.set_ylabel('Porcentagem do Tempo (%)')
    ax.set_xlabel('Tipo')
    ax.set_title(f'Consumo de Tempo por Tipo e Função (Num Generations={num_generations_fixo}, Size={size_fixo})')

    # Exiba o gráfico
    plt.show()

if __name__ == "__main__":
    criar_grafico(sys.argv)