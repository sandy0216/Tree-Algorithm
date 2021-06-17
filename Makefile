CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O3 -Wall
NVCC    := nvcc
#NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true -fopenmp

LIB     := -lcufft -lcudart 
NVFLAGS := -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true -lgsl -lgslcblas -lm

TARGET := ./tree
OBJ    := ./lib/heap.o ./lib/init.o ./lib/print_tree.o ./lib/tool_tree.o ./lib/tool_tree_gpu.o ./lib/create_tree.o ./lib/create_tree_gpu.o ./lib/force.o ./lib/tool_main.o ./lib/main.o

$(TARGET):$(OBJ)
	$(NVCC) $(OBJ) $(NVFLAGS) -o $(TARGET)

#%.o: %.c
#	$(NVCC) -c -o $< $@

./lib/main.o : ./src/main.cu 
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/init.o : ./src/init.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/tool_tree.o : ./src/tool_tree.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/tool_tree_gpu.o : ./src/tool_tree_gpu.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/create_tree.o : ./src/create_tree.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/create_tree_gpu.o : ./src/create_tree_gpu.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/print_tree.o : ./src/print_tree.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/force.o : ./src/force.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/tool_main.o : ./src/tool_main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/heap.o : ./src/heap.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@


#./lib/def_node.o : ./src/def_node.cu
#	$(NVCC) $(NVFLAGS) -c $< -o $@
	


clean:
	rm -f $(TARGET) $(OBJ) *.o ./output/*
