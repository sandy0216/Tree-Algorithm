CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O3 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true
LIB     := -lcufft -lcudart

BIN2    := fft_cuFFT

all: $(BIN2) 


$(BIN2): fft_cuFFT.cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) $< $(LIB)


clean:
	rm -f $(BIN1) $(BIN2) $(BIN3) *.o
