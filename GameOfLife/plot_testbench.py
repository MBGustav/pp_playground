import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def print_graphic(file_name):
# Read the CSV file
    data = pd.read_csv(file_name, delimiter=",")

    # Sort the data based on 'size'
    # data = data.sort_values(by='size')

    # Get unique function names
    platforms = data['type'].unique()
    functions = data['function'].unique()

     # Get unique function names
    platforms = data['type'].unique()
    functions = data['function'].unique()
    openMP_max = data.groupby('type').get_group('openMP')['time_ms'].max()
    openMP_min = data.groupby('type').get_group('openMP')['time_ms'].min()
    serial_max = data.groupby('type').get_group('serial')['time_ms'].max()
    serial_min = data.groupby('type').get_group('serial')['time_ms'].min()
    size_serial = data.groupby('type').get_group('serial')['size'].max()
    size_OMP = data.groupby('type').get_group('openMP')['size'].max()
    size_cuda = data.groupby('type').get_group('cuda_v2')['size'].max()
    
    m_omp =  (openMP_max - openMP_min) / size_OMP
    m_serial =  (serial_max - serial_min) / size_serial 
    new_OMP = m_omp * size_cuda
    new_serial = m_serial * size_cuda

    data.loc[len(data)] = ['serial', 100, size_cuda, 'Total Time Execution', new_serial]
    data.loc[len(data)] = ['openMP', 100, size_cuda, 'Total Time Execution', new_OMP] 


    m_omp =  (openMP_max - openMP_min) / size_OMP
    m_serial =  (serial_max - serial_min) / size_serial 
    new_OMP = m_omp * size_cuda
    new_serial = m_serial * size_cuda

    data.loc[len(data)] = ['serial', 100, size_cuda, 'Total Time Execution', new_serial]
    data.loc[len(data)] = ['openMP', 100, size_cuda, 'Total Time Execution', new_OMP] 

    # data = data.query('time_ms > 0 and time_ms < 1000')
    # data = data.query('size > 0 and size < 1600')
    grp = data.groupby(['type','function','size', 'NumberOfGenerations'])
    grp = grp.agg({'time_ms': ['mean', 'std', 'min', 'max']})

    plt.figure(figsize=(10, 6))

    for func in platforms:
        func_result = grp.loc[func]  # Filtra o resultado para a função específica
        plt.errorbar(func_result.index.get_level_values('size'), func_result['time_ms']['mean'], 
                     label=func, 
                    linestyle='-', capsize=5, capthick=2)

    plt.xlabel('tamanho da Janela')
    plt.ylabel('Execution Time (ms)')
    plt.title('Comparação por tamanho de Janela ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("bench-gemm")
    plt.show()


if len(sys.argv) >1:
    for i in range(1, len(sys.argv)):
        print_graphic(sys.argv[i])
else:
    print("Usage:", sys.argv[0],"<file_name>")
    print("In this file, the script makes a comparative with only time execution.")
    print("For more detailed benchmark, with a third parameter (size, for example), use: ./testbench-compare")

