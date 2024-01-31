

# COMPILERS USED
CC = gcc
NVCC = nvcc
NSYS=nsys profile

# COMPILE FLAGS
CXX_FLAGS  = -g -Wall -Werror
NVXX_FLAGS = -g -G
NSYS_FLAGS =--stats=true --force-overwrite=true
CXX_VECTORIZATION = $(CXX_FLAGS) -O3  -fopenmp -mavx

# OPTIONAL FLAGS(UNCOMMENT IF YOU WANT FOR DEBUG)
OPT_FLAGS =
# OPT_FLAGS += -D_DISPLAY_GAME
# OPT_FLAGS += -D_DEBUG_PER_STEP

PROJECT = Gol
ARGS = 100

# FOLDERS
SERIAL = serial
OPENMP = openMP
INCLUDE_DIR = ./common
BINARY_DIR  = ./bin
OUTPUT_DIR  = ./output

# FOLDER CREATION
bin_guard=@mkdir -p $(BINARY_DIR)
out_guard=@mkdir -p $(OUTPUT_DIR)

# COMPILING PARAMETERS
ifdef SIZE
	OPT_FLAGS +=-DWidth=$(SIZE) -DHeight=$(SIZE)
endif

ifdef BLOCK_SIZE
	OPT_FLAGS +=-DBLOCK_SIZE=$(BLOCK_SIZE)
endif

ifdef NUM_STREAMS
	OPT_FLAGS +=-DNUM_STREAMS=$(NUM_STREAMS)
endif


.PHONY: all
all: serial openMP cuda_v0 cuda_v1 cuda_v2

.PHONY: run_gpu
run_gpu: cuda_v0_run cuda_v1_run cuda_v2_run

.PHONY: run_all
run_all: serial_run openMP_run run_gpu

.PHONY: serial
serial: 
	$(bin_guard)
	$(CC) $(CXX_FLAGS) $(OPT_FLAGS) -I $(INCLUDE_DIR)  $(SERIAL)/$(PROJECT).c -o $(BINARY_DIR)/$(SERIAL).out

.PHONY: openMP
openMP: 
	$(bin_guard)
	$(CC) $(CXX_VECTORIZATION) $(OPT_FLAGS) -I $(INCLUDE_DIR) $(OPENMP)/$(PROJECT).c -o $(BINARY_DIR)/$(OPENMP).out

.PHONY: cuda_v0
cuda_v0: 
	$(bin_guard)
	$(NVCC) $(NVXX_FLAGS) $(OPT_FLAGS) -I $(INCLUDE_DIR) $@/$(PROJECT).cu -o $(BINARY_DIR)/$@.out

.PHONY: cuda_v1
cuda_v1: 
	$(bin_guard)
	$(NVCC) $(NVXX_FLAGS) $(OPT_FLAGS) -I $(INCLUDE_DIR) $@/$(PROJECT).cu -o $(BINARY_DIR)/$@.out

.PHONY: cuda_v2
cuda_v2: 
	$(bin_guard)
	$(NVCC) $(NVXX_FLAGS) $(OPT_FLAGS) -I $(INCLUDE_DIR) $@/$(PROJECT).cu -o $(BINARY_DIR)/$@.out


serial_run: 
	$(out_guard)
	./$(BINARY_DIR)/$(SERIAL).out $(ARGS) 

openMP_run: 
	$(out_guard)
	./$(BINARY_DIR)/$(OPENMP).out $(ARGS) 

cuda_v0_run: 
	$(out_guard)
	./$(BINARY_DIR)/cuda_v0.out $(ARGS)
	$(NSYS) $(NSYS_FLAGS) -o $(OUTPUT_DIR)/report-cuda_v0 ./$(BINARY_DIR)/cuda_v0.out $(ARGS) > /dev/null

cuda_v1_run: 
	$(out_guard)
	./$(BINARY_DIR)/cuda_v1.out $(ARGS)
	$(NSYS) $(NSYS_FLAGS) -o $(OUTPUT_DIR)/report-cuda_v1 ./$(BINARY_DIR)/cuda_v1.out $(ARGS) > /dev/null

cuda_v2_run: 
	$(out_guard)
	./$(BINARY_DIR)/cuda_v2.out $(ARGS)
	$(NSYS) $(NSYS_FLAGS) -o $(OUTPUT_DIR)/report-cuda_v2 ./$(BINARY_DIR)/cuda_v2.out $(ARGS) > /dev/null



clean:
	rm -rf $(OUTPUT_DIR)/* $(BINARY_DIR)/*


