
//Facilitando o uso de GPUcomplexo
struct GPUComplex
{
    float r,i;
    __device__ GPUComplex(float a,float b): r(a), i(b){}
    __device__ float mag() {return r*r+i*i;}
    __device__ GPUComplex operator*(const GPUComplex &a){
        return GPUComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ GPUComplex operator+(const GPUComplex &a){
        return GPUComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y, int DIM) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);
    
    GPUComplex c(-0.8, 0.156);
    GPUComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;

        //Verifica Divergencia
        if (a.mag() > 1000)
            return 0;
    }
    return 1;
}

 __global__ void Julia_Kernel(unsigned char* ptr, int dim){


    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int offset = x + y* gridDim.x;
    if(x < dim && y < dim){
        int Jval = julia( x, y, dim);
        ptr[offset*3 + 0] = 255 * Jval;
        ptr[offset*3 + 1] = 150;
        ptr[offset*3 + 2] = 0;
    }
}

void JuliaFilter( unsigned char *ptr, int dim ){

    dim3 ThreadsPerBlock(16,16);//total Threads : 16*16 = 256
    dim3 Grid((dim+16-1)/16,(dim+16-1)/16);

    int sizeof_ptr = dim*dim * 3 * sizeof(unsigned char);
    unsigned char *d_ptr;
    
    //Alloc memory for ptr -- > image vector && copy host val
    cudaMalloc(&d_ptr, sizeof_ptr);
    cudaMemcpy(&d_ptr, ptr, sizeof_ptr, cudaMemcpyHostToDevice);

    //Kernel submission
    Julia_Kernel<<<Grid, ThreadsPerBlock>>>(d_ptr, dim);


    cudaMemcpy(ptr, d_ptr, sizeof_ptr, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


}

 
