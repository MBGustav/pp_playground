CC=g++
NVCC=nvcc
EXEC=JF
FLAGS = -O3 -Werror



#CUDA execution 
GPU_EXEC=$(EXEC)-cuda

# openMP execution
MP_EXEC=$(EXEC)-openMP
MP_FLAGS= $(FLAGS) -fopenmp -DOPENMP -DIM



build_all: build_cuda build_openmp build_serial
	@echo "Building Completed"

build_cuda: 
	$(NVCC) -o $(GPU_EXEC) $(GPU_EXEC).cu

build_openmp:
	$(CC) -o $(MP_EXEC) $(MP_FLAGS) $(MP_EXEC).cpp

build_serial:
	$(CC) -o $(EXEC) $(FLAGS) -DSERIAL $(EXEC).cpp


run_cuda: build_cuda
	./$(GPU_EXEC)

run_openmp: build_openmp
	./$(MP_EXEC)

run_serial: build_serial
	./$(EXEC) 


clean:
	rm -rf $(EXEC) $(GPU_EXEC) $(MP_EXEC) $(INPUT_GEN) julia_* *.o

