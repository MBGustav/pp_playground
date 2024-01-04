# Uso de PIPES com oneapi

Neste Exemplo, fazemos o uso de pipelines para realização de produto vetorial, onde:
$$
        C_i = A_i \cdot B_i
$$

É utilizado apenas um pipe ao todo, do qual é feito uma ponte entre o host e o Device.

Para execução, use: 
```shell
$ icpx -fsycl -fintelfpga -DFPGA_EMULATOR main.cpp -o main.fpga.emu; 
$ ./main.fpga.emu
```
