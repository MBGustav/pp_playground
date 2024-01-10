#COMPILERS
CXX= gcc
NVXX=nvcc
NSYS= nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

# COMPILERS FLAGS
CXXFLAGS= -O3 -fopenmp -mavx -Wall
NVXXFLAGS=

EXECUTABLE=odd-even

SRC_INPUT=input

SRC_GPU=$(wildcard ./*.cu*)
INC_DIR=include
STD_INPUT=1024

BIN=bin
REPORT=report
bin_guard=@mkdir -p $(BIN)
report_guard=@mkdir -p $(REPORT)


all: input gpu_v1 serial

serial: input
	$(bin_guard)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) $(EXECUTABLE).c -o $(BIN)/$(EXECUTABLE).cpu

gpu_v0:
	$(bin_guard)
	$(NVXX) $(NVXXFLAGS) -I$(INC_DIR) odd-even.cu -o $(BIN)/$(EXECUTABLE)_v0.gpu
gpu_v1:
	$(bin_guard)
	$(NVXX) $(NVXXFLAGS) -I$(INC_DIR) odd-even_v1.cu -o $(BIN)/$(EXECUTABLE).gpu


run_serial: $(BIN)/$(EXECUTABLE).cpu
	$(bin_guard) 
	./$< $(BIN)/$(SRC_INPUT).in


run_gpu_v1: $(BIN)/$(EXECUTABLE).gpu
	$(bin_guard) 
	./$< $(BIN)/$(SRC_INPUT).in

input: $(SRC_INPUT).c
	$(bin_guard)
	$(CXX) $(CXXFLAGS) $(SRC_INPUT).c  -o $(SRC_INPUT).out

profile: $(BIN)/$(EXECUTABLE).gpu
	$(report_guard)
	$(NSYS) $(NSYSFLAGS) -o $(REPORT)/report ./$<

clean:
	rm $(BIN)/* *.out *.in
