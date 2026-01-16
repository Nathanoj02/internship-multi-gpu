# Compiler settings
NVCC := nvcc
CXX := g++

# Project directories
SRC_DIR := src
INC_DIRS := inc ../utils
OBJ_DIR := obj
BIN_DIR := bin

# Target executable
TARGET := $(BIN_DIR)/gemm

# Source files
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Object files
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.cu.o,$(CU_SOURCES))
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.cpp.o,$(CPP_SOURCES))
OBJECTS := $(CU_OBJECTS) $(CPP_OBJECTS)

# Compiler flags
NVCC_FLAGS := $(addprefix -I,$(INC_DIRS)) -O3 -gencode arch=compute_75,code=sm_75 #-gencode arch=compute_89,code=sm_89
CXX_FLAGS := $(addprefix -I,$(INC_DIRS)) -O3 -std=c++11

# CUDA libraries
CUDA_LIBS := -lcudart

# Default target
.PHONY: all
all: $(TARGET)

# Create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Link the executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUDA_LIBS)

# Compile CUDA source files
$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run the program
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Print variables for debugging
.PHONY: info
info:
	@echo "CUDA Sources: $(CU_SOURCES)"
	@echo "C++ Sources: $(CPP_SOURCES)"
	@echo "CUDA Objects: $(CU_OBJECTS)"
	@echo "C++ Objects: $(CPP_OBJECTS)"
	@echo "Target: $(TARGET)"
