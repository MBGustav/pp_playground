# LIB IMPLEMENTATION FOR NVIDIA GPU OPERATION 


In this project, I'm developing a cuda (maybe associated with openMP afterwards) that deals with Matricial operations. Taking this into consideration, the following operations are expected to be used. 

## Why to implement this?

The main idea is, by the host associated to GPU Nvidia properties, the library chooses between the best overhead and in the future deal with Queue executions (EventHandler).


## Plans and Observations? 
|done?|Oper.|Description|
|:-:|:-:|:-:|
|-|add/sub|Sum/Sub Operation      |
|+|scal   |scalar Operation       |
|+|gemm   |gemm-like Operation    |
|-|det    |get Matrix determinant |
|-|inv    |Inverse of a matrix    |
|-|Async  |Asynch Changes Refresh |





## BrainStorm
- Should I create a Matrix that contains multiples sub-matrices to handle with tiling in certain cases?
- How to develop an EventHandler to know when device must be synch with host?
- How to adquire GPU info to auto-infer max threads, max blocks and number of Gpu's? Cuda Properties
- Should I implement and info collection to get the best markups to select in-place operation and offload ? 
- How to associate threads (from c++) to make async operations CPU/GPU?
- How to increment diferent types of implementations, such as  SYCL/OpenCL?