CUDIR   := /usr/local/nvidia

CC      := g++
CFLAGS  := -O3 -Wall
NVCC    := nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_61 -code=sm_61 -Xptxas -v -rdc=true
LIB     := -lcufft -lcudart

TARGET := ./tree
OBJ    := ./lib/init.o ./lib/print_tree.o ./lib/tool_tree.o ./lib/create_tree.o ./lib/force.o ./lib/main.o

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
./lib/create_tree.o : ./src/create_tree.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/print_tree.o : ./src/print_tree.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
./lib/force.o : ./src/force.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@
#./lib/def_node.o : ./src/def_node.cu
#	$(NVCC) $(NVFLAGS) -c $< -o $@
	


clean:
	rm -f $(TARGET) $(OBJ) *.o