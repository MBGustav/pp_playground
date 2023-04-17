CC=gcc
NVCC=nvcc
FLAGS = -O3 -Wextra # -DNDEBUG 
EXEC=maxSum

#CUDA execution 
GPU_EXEC=$(EXEC)-cuda
GPU_FLAGS= -DNDEBUG

# openMP execution
MP_EXEC=$(EXEC)-openMP
MP_FLAGS= $(FLAGS) -fopenmp

#generator input file
INPUT_GEN=input-generator
N_POINTS=1000
N_LABELS=50

input:
	$(CC) -o $(INPUT_GEN) $(FLAGS) $(INPUT_GEN).c
	./$(INPUT_GEN) $(N_POINTS) $(N_LABELS)

build_all: build_cuda build_serial
	@echo "Building Completed"

build_cuda: 
	$(NVCC) -o $(GPU_EXEC) $(GPU_FLAGS) $(GPU_EXEC).cu

# build_openmp:
# 	$(CC) -o $(MP_EXEC) $(MP_FLAGS) $(MP_EXEC).c 

build_serial:
	$(CC) -o $(EXEC) $(FLAGS) $(EXEC).c 


run_cuda: build_cuda
	./$(GPU_EXEC) < input.txt

run_openmp: build_openmp
	./$(MP_EXEC)  < input.txt

run_serial: build_serial
	./$(EXEC)  < input.txt


clean:
	rm -rf $(EXEC) $(GPU_EXEC) $(MP_EXEC) $(INPUT_GEN) input.txt *.o

