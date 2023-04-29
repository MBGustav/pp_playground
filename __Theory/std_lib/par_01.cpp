#include <iostream> 
#include <execution> 
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <execution>

namespace ex = std::execution;


void using_lambda(auto pol, std::vector<int> &input, std::vector<int> x, int a){
    // Declarando funcao lambda
    auto f = [x = x.data(), size = x.size()](int i) {
        int acc = 0;
        for (int j = 0; j < size; j++) {
            acc += j * x[j];
        }
        return acc * i;
    };


    std::for_each(pol,          //tipo de execução
    begin(input), end(input),   //inicio-entrada, final-entrada 
    [&] (int &el){              //valor, elemento de entrada-saida
        el = f(el);             //funcao-lamda
    });    
}

enum exec {seq, par, par_unseq};

int main(int argc, char **argv){
    if(argc > 3 || argc ==1){
        std::cout << "Invalid input\n";
        return EXIT_FAILURE;
    }
    srand(777);
    int tp = atoi(argv[1]);
    const int size = atoi(argv[2]);

    std::vector<int> v(size), x = {1,2,3};
    
    for(int i=0; i < size; i++)
        v[i] = rand() %size;
    
    std::cout << "Execucao do tipo: ";
    switch (tp){
        case par: {
            std::cout << "par\n";
            using_lambda(ex::par, v, x, 2);
            break;
            }
        case par_unseq: {
            std::cout << "par_unseq\n";
            using_lambda(ex::par_unseq, v, x, 2);
            break;
            }
        default:{
            std::cout << "seq\n";
            using_lambda(ex::seq, v, x, 2);
            break;
            }
    }


    return 0;
}

