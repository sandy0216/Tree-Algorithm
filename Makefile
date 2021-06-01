CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O3 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true
LIB     := -lcufft -lcudart

TARGET := ./tree
OBJ    := ./lib/main.o ./lib/init.o

$(TARGET):$(OBJ)
	$(NVCC) $^ $(NVFLAGS) -o $(TARGET) 

./lib/main.o : ./src/main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/init.o : ./src/init.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
	


clean:
	rm -f $(TARGET) $(OBJ) *.o
