# Compiler settings
NVCC := nvcc
CXX := mpicxx

# Project directories
SRC_DIR := src
INC_DIRS := inc ../utils
OBJ_DIR := obj
BIN_DIR := bin

# Target executable
TARGET := $(BIN_DIR)/reduction

# Auto-detect MPI include path for NVCC
MPI_INCLUDE := $(shell $(CXX) --showme:incdirs)
NVCC_MPI_FLAGS := $(addprefix -I,$(MPI_INCLUDE))

# Source files
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Object files
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SOURCES))
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.cpp.o,$(CPP_SOURCES))
OBJECTS := $(CU_OBJECTS) $(CPP_OBJECTS)

# Compiler flags
NVCC_FLAGS := $(addprefix -I,$(INC_DIRS)) $(NVCC_MPI_FLAGS) -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89
CXX_FLAGS := $(addprefix -I,$(INC_DIRS)) -O3 -std=c++11

# CUDA libraries
CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart

# Default target
.PHONY: all
all: $(TARGET)

# Create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Link the executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(CUDA_LDFLAGS)

# Compile CUDA source files
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run the program
NUM_GPUS := $(shell nvidia-smi -L | wc -l)
NP ?= $(NUM_GPUS)
.PHONY: run
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

# Print variables for debugging
.PHONY: info
info:
	@echo "MPI Include Path detected: $(MPI_INCLUDE)"
	@echo "Target: $(TARGET)"
