MPICC = mpicc

TARGET = bin/latency
SRCS = src/latency.c

NP = 2

all: $(TARGET)

$(TARGET): $(SRCS)
	mkdir -p bin
	$(MPICC) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@mpirun -np $(NP) $(TARGET)

.PHONY: all clean run
