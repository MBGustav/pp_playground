

# COMPILERS USED
CC = gcc
NVCC = nvcc

# COMPILE FLAGS
CXX_FLAGS  = -g -Wall -Werror
NVXX_FLAGS = -g -G

CXX_VECTORIZATION = $(CXX_FLAGS) -O3  -fopenmp -mavx

# OPTIONAL FLAGS(UNCOMMENT IF YOU WANT)
OPT_FLAGS =
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
all: openMP cuda_v0 cuda_v1 serial
run_all: serial_run openMP_run cuda_v0_run

.PHONY: serial
serial: 
	@echo $(PROJECT_FILES)
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


serial_run: serial 
	$(out_guard)
	./$(BINARY_DIR)/$(SERIAL).out $(ARGS) > $(OUTPUT_DIR)/$<.txt

openMP_run: openMP 
	$(out_guard)
	./$(BINARY_DIR)/$(OPENMP).out $(ARGS) > $(OUTPUT_DIR)/$<.txt

cuda_v0_run: cuda_v0
	$(out_guard)
	./$(BINARY_DIR)/$<.out $(ARGS) > $(OUTPUT_DIR)/$<.txt

cuda_v1_run: cuda_v1
	$(out_guard)
	./$(BINARY_DIR)/$<.out $(ARGS) > $(OUTPUT_DIR)/$<.txt

clean:
	rm -rf $(OUTPUT_DIR)/* $(BINARY_DIR)/*


