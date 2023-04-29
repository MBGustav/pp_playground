#include <iostream> 
#include <execution> 
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <mutex>
namespace execution = std::execution; 


int ParSeq(std::vector<int>& v){
    int sum = 0;
    std::mutex m;

    std::for_each(execution::par, v.begin(), v.end(), [&](int i){
        std::lock_guard<std::mutex> lock(m);
        sum += i;
    });
     
    return sum;    
}

int ParUnseq_v1(std::vector<int> v){
    std::atomic<int> sum{0};
    std::for_each(execution::par_unseq, v.begin(), v.end(), 
    [&](int i){ sum+=i; });
    
    return sum.load();    
}
int ParUnseq_v2(std::vector<int> v){
    //# std::atomic<int> global{0};
    int global= std::reduce(execution::par_unseq, v.begin(), v.end(), std::plus{});
    return global;

}

int main(){
    int size = 10000;
    std::vector<int> v(size);  
    for(int i=0; i < size; i++){
        v[i] = rand() %1000;
    }
    
    std::cout <<"ParSeq     "<< ParSeq(v)<< std::endl;
    std::cout <<"ParUnseq_v1"<< ParUnseq_v1(v)<< std::endl;
    std::cout <<"ParUnseq_v2:"<< ParUnseq_v2(v)<< std::endl;
    
    

}
