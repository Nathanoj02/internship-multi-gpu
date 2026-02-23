MPICC = mpicc

TARGET = bin/cannon
SRCS = src/cannon.c

NP = 64

all: $(TARGET)

$(TARGET): $(SRCS)
	mkdir -p bin
	$(MPICC) -o $(TARGET) $(SRCS) -lm

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@mpirun --oversubscribe -np $(NP) $(TARGET)

.PHONY: all clean run
