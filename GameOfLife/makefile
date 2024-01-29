

# COMPILERS USED
CC = gcc
NVCC = nvcc
NSYS=nsys profile

# COMPILE FLAGS
CXX_FLAGS  = -g -Wall -Werror
NVXX_FLAGS = -g -G
NSYS_FLAGS =--stats=true --force-overwrite=true


CXX_VECTORIZATION = $(CXX_FLAGS) -O3  -fopenmp -mavx

# OPTIONAL FLAGS(UNCOMMENT IF YOU WANT)
OPT_FLAGS =
# OPT_FLAGS += -D_DISPLAY_GAME
# OPT_FLAGS += -D_DEBUG_PER_STEP


# FOLDERS
SERIAL = serial
OPENMP = openMP

INCLUDE_DIR = ./common
BINARY_DIR  = ./bin
OUTPUT_DIR  = ./output

bin_guard=@mkdir -p $(BINARY_DIR)
out_guard=@mkdir -p $(OUTPUT_DIR)

PROJECT = Gol

ARGS = 100


.PHONY: all
all: serial openMP cuda_v0 cuda_v1 cuda_v2
run_all: serial_run openMP_run cuda_v0_run cuda_v1_run cuda_v2_run

.PHONY: serial
serial: 
	$(bin_guard)
	$(CC) $(CXX_FLAGS) -I $(INCLUDE_DIR) $(OPT_FLAGS) $(SERIAL)/$(PROJECT).c -o $(BINARY_DIR)/$(SERIAL).out

.PHONY: openMP
openMP: 
	$(bin_guard)
	$(CC) $(CXX_VECTORIZATION) -I $(INCLUDE_DIR) $(OPT_FLAGS) $(OPENMP)/$(PROJECT).c -o $(BINARY_DIR)/$(OPENMP).out

.PHONY: cuda_v0
cuda_v0: 
	$(bin_guard)
	$(NVCC) $(NVXX_FLAGS) -I $(INCLUDE_DIR) $(OPT_FLAGS) $@/$(PROJECT).cu -o $(BINARY_DIR)/$@.out

.PHONY: cuda_v1
cuda_v1: 
	$(bin_guard)
	$(NVCC) $(NVXX_FLAGS) -I $(INCLUDE_DIR) $(OPT_FLAGS) $@/$(PROJECT).cu -o $(BINARY_DIR)/$@.out

.PHONY: cuda_v2
cuda_v2: 
	$(bin_guard)
	$(NVCC) $(NVXX_FLAGS) -I $(INCLUDE_DIR) $(OPT_FLAGS) $@/$(PROJECT).cu -o $(BINARY_DIR)/$@.out


serial_run: serial 
	$(out_guard)
	./$(BINARY_DIR)/$(SERIAL).out $(ARGS) 

openMP_run: openMP 
	$(out_guard)
	./$(BINARY_DIR)/$(OPENMP).out $(ARGS) 

cuda_v0_run: cuda_v0
	$(out_guard)
	$(NSYS) $(NSYS_FLAGS) -o $(OUTPUT_DIR)/report-$< ./$(BINARY_DIR)/$<.out $(ARGS) 

cuda_v1_run: cuda_v1
	$(out_guard)
	$(NSYS) $(NSYS_FLAGS) -o $(OUTPUT_DIR)/report-$< ./$(BINARY_DIR)/$<.out $(ARGS) 

cuda_v2_run: cuda_v2
	$(out_guard)
	$(NSYS) $(NSYS_FLAGS) -o $(OUTPUT_DIR)/report-$< ./$(BINARY_DIR)/$<.out $(ARGS) 


clean:
	rm -rf $(OUTPUT_DIR)/* $(BINARY_DIR)/*


