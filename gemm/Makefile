# Compiler settings
NVCC := nvcc
CXX := g++

# Project directories
SRC_DIR := src
INC_DIRS := inc ../utils
OBJ_DIR := obj
BIN_DIR := bin

# Target executables
TARGET := $(BIN_DIR)/gemm
TARGET_WARP := $(BIN_DIR)/gemm_warp_tiling

# Source files (excluding main files)
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/kernels/*.cu)
CPP_SOURCES := $(filter-out $(SRC_DIR)/main.cpp $(SRC_DIR)/main_warp_tiling.cpp, $(wildcard $(SRC_DIR)/*.cpp))

# Object files
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SOURCES))
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.cpp.o,$(CPP_SOURCES))
COMMON_OBJECTS := $(CU_OBJECTS) $(CPP_OBJECTS)

# Main object files
MAIN_OBJ := $(OBJ_DIR)/main.cpp.o
MAIN_WARP_OBJ := $(OBJ_DIR)/main_warp_tiling.cpp.o

# Compiler flags
NVCC_FLAGS := $(addprefix -I,$(INC_DIRS)) -O3 -gencode arch=compute_75,code=sm_75 #-gencode arch=compute_89,code=sm_89
CXX_FLAGS := $(addprefix -I,$(INC_DIRS)) -O3 -std=c++11

# CUDA libraries
CUDA_LIBS := -lcudart

# Default target
.PHONY: all
all: $(TARGET)

# Build both executables
.PHONY: both
both: $(TARGET) $(TARGET_WARP)

# Create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Link the main executable
$(TARGET): $(COMMON_OBJECTS) $(MAIN_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUDA_LIBS)

# Link the warp tiling executable
$(TARGET_WARP): $(COMMON_OBJECTS) $(MAIN_WARP_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUDA_LIBS)

# Force rebuild when definitions change
DEFINITIONS := inc/definitions.hpp

# Compile CUDA source files
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(DEFINITIONS) | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp $(DEFINITIONS) | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run the main program
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Run the warp tiling benchmark
.PHONY: run-warp
run-warp: $(TARGET_WARP)
	./$(TARGET_WARP)

# Build warp tiling executable (for autotuner)
.PHONY: warp
warp: $(TARGET_WARP)

# Print variables for debugging
.PHONY: info
info:
	@echo "CUDA Sources: $(CU_SOURCES)"
	@echo "C++ Sources: $(CPP_SOURCES)"
	@echo "Common Objects: $(COMMON_OBJECTS)"
	@echo "Main Target: $(TARGET)"
	@echo "Warp Target: $(TARGET_WARP)"
