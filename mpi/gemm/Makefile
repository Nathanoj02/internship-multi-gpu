MPICC = mpicc

TARGET = bin/gemm
SRCS = src/gemm.c

NP = 4

all: $(TARGET)

$(TARGET): $(SRCS)
	mkdir -p bin
	$(MPICC) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@mpirun -np $(NP) $(TARGET)

.PHONY: all clean run
