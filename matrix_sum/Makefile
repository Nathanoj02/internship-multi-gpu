# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -Iinc
LDFLAGS =

# Directories
SRC_DIR = src
INC_DIR = inc
BIN_DIR = bin
OBJ_DIR = obj

# Target executable name
TARGET = $(BIN_DIR)/matrix_sum

# Find all C++ source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Generate object file names
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create directories if they don't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Rebuild everything
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)

# Show variables (for debugging)
debug:
	@echo "SRCS: $(SRCS)"
	@echo "OBJS: $(OBJS)"
	@echo "TARGET: $(TARGET)"

.PHONY: all clean rebuild run debug