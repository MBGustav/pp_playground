#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>



// Defining two pipes - name mangling
class FirstPipe;
class SecndPipe;

// Kernel Operations realized - mangling
class SendToDevice;
class SendToHost;


#define FIRST_PIPE_CAPACITY (8)
#define SECOND_PIPE_CAPACITY (8)
#define MAX_LEN (124)

using namespace sycl;
using std::cout;

// Just to make things easier
typedef std::vector<float> Fvector;

using InputPipe = ext::intel::experimental::pipe<
    FirstPipe, 
    float, 
    FIRST_PIPE_CAPACITY
>;

using OutputPipe = ext::intel::experimental::pipe<
    SecndPipe, 
    float, 
    SECOND_PIPE_CAPACITY
>;



int main(int argc, char **argv)
{
    int size;
    if(argc == 2)
        size = atoi(argv[1]);
    else 
        size = MAX_LEN;

    //Device selection
    //We will explicitly compile for the FPGA_EMULATOR, CPU_HOST, or FPGA
    #if defined(FPGA_SIMULATOR)
     auto selector = sycl::ext::intel::fpga_simulator_selector_v;
    #elif defined(FPGA_HARDWARE)
     auto selector = sycl::ext::intel::fpga_selector_v;
    #else //FPGA_EMULATOR
     auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    #endif

    auto property_list = sycl::property_list{
                sycl::property::queue::enable_profiling()};

    sycl::queue queue = sycl::queue(selector, property_list);

    cout << "Running on: " 
         << queue.get_device().get_info<sycl::info::device::name>().c_str()
         << std::endl;
         

    cout << "With size Input of len" << size << std::endl;

    Fvector vecA(size);
    Fvector vecB(size);
    Fvector output_array(size);


    // Generating random values
    for(int i =0; i < size; i++){
        vecA[i] = (float) rand() / ( (float) RAND_MAX) * 10.0f;
        vecB[i] = (float) rand() / ( (float) RAND_MAX) * 10.0f;

    }
    
    //Create Buffers
    buffer input_b1(vecA);
    buffer input_b2(vecB);
    buffer output_buf(output_array);

    // Host emission
    auto emit_ev = queue.submit([&] (handler &h){
        accessor acc1(input_b1, h, read_only);
        accessor acc2(input_b2, h, read_only);
        int size_acc = acc1.size();
        // Insert each value alternatively -> arr1[0] -> arr2[0] -> ...
        h.single_task<SendToHost> ([=](){
            for(int i =0; i < size_acc ; i++){
                InputPipe::write(acc1[i]);
                InputPipe::write(acc2[i]);
            }
        });
    });

    emit_ev.wait();

    auto recv_ev = queue.submit([&] (handler &h){
        accessor acc1(output_buf, h, write_only);
        int size_acc = acc1.size();
        h.single_task<SendToDevice> ([=](){

            for(int i =0; i < size_acc ; i++){

            // Make two reads --> out = arr1* arr2
                float a = InputPipe::read();
                float b = InputPipe::read();
                float res =  a*b;
                acc1[i]  = res;
            }
        });
    });

    recv_ev.wait();

    // Checking results :
    for(int i=0; i< size; i++)
    {
        float res = vecA[i] * vecB[i];
        if(fabs(output_array[i] - res)>0.001f)
        {
            cout << "Wrong Result! \n";
            cout << "\t " << output_array[i] <<" != " 
                 << res << "\n";
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Finished, all results are correct!"<<std::endl;

    return 0;
}