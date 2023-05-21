#include <iostream> 
#include <execution> 
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
namespace execution = std::execution; 


int data_race(auto policy, std::vector<int> v){

    int sum = 0.;
    std::for_each(policy, v.begin(), v.end(), [&](int i){
        sum+= i;
    });
    return sum;    
}


int main(){
    int size = 10000;
    std::vector<int> v(size);  
    for(int i=0; i < size; i++){
        v[i] = rand() %1000;
    }
    
    std::cout <<"seq:      "<< data_race(execution::seq, v)<< std::endl;
    std::cout <<"unseq:    "<< data_race(execution::unseq, v)<< std::endl;
    std::cout <<"par_unseq:"<< data_race(execution::par_unseq, v)<< std::endl;
    std::cout <<"par:      "<< data_race(execution::par, v)<< std::endl;
    
    

}
