
## Diferencial de Implementação
Considerando a implementação _naive_ anterior, onde se tem apenas o desenvolvimento do kernel para _offload_ (por meio do lambda `single_task`), foi incluido também no desenvolvimento o uso de _Shift Registers_, evitando multiplos acessos a um registrador. Isso possibilita menor numero de ciclos de clock para cálcular o valor resultante de acumulação. Melhores detalhes podem ser vistos no recorte abaixo:


```cpp

''1.0 Declara ao Compilador independencia de dados (ivdep)''
        [[intel::ivdep]] 
''1.1: Uso de apenas um loop para toda a execução''
        for (int idx = 0; idx < kArows * kBcols; ++idx) {
            int i = (int) idx / kBcols;
            int j = (int) idx % kBcols;

''2.0: Uso de _copy_of_dprod_ como um shift register ''
            #pragma unroll
            for(int i= 0; i < MAX_SIZE_TEMPORARY_BUFFER; i++)
                copy_of_dprod[i] = 0.0f;

            for (int k = 0; k < kCommon; k++){
                data_t current_dprod = 
                    dev_A[kCommon * i + k] * dev_B[kBcols * k + j];

                copy_of_dprod[MAX_SIZE_TEMPORARY_BUFFER] =
                    copy_of_dprod[0] + current_dprod;

                #pragma unroll (MAX_SIZE_TEMPORARY_BUFFER)
                for(int pip= 0; pip < MAX_SIZE_TEMPORARY_BUFFER; pip ++)
                    copy_of_dprod[pip] = copy_of_dprod[pip+1];
            }
''3.0: Uso de variavel local para acumular resultado''
            data_t acc = 0.0f;
            #pragma unroll (MAX_SIZE_TEMPORARY_BUFFER)
            for(int ii=0; ii < MAX_SIZE_TEMPORARY_BUFFER; ii++)
                acc += copy_of_dprod[ii];
''3.1: Reducao de Acesso a memoria por meio de apenas uma escrita ''
            dev_C[i*kArows + j] = acc;
        }
```

Esta implementação reduziu o numero total de clocks necessarios para realização do processo esperado.


