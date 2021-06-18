# Tree-Algorithm

This is README file for the final project of 2021 CUDA parallel programming course
Author : Huai-Hsuan Chiu
Date   : June, 2021

=====
DIRECTORY:

input/	place to record input data
output/	place to record output data
inc/	header(*.h) of each file
	param.h		: One can change the boxsize, number of particles, block size & grid size(for GPU) here.
lib/	output(*.o) of the complie
src/	source codes
	main.cu		: main functions
	init.cu		: Handle the creation of the initial condition
	create_tree.cu	: CPU version of the tree creation
	tool_tree.cu	: Some tools of tree creation
	(Blank node creation, region checker, recursive function of the finest grid)
	print_tree.cu	: Print out the tree for checking	
	force.cu	: CPU version of force & potential calculator
	tool_main.cu	: Some tools of the main function
	(Boundary checker, remove of the particle, update the position&velocity of the particle)
	create_tree_gpu.cu	: GPU version of the tree creation
	tool_tree_gpu.cu	: Tools of GPU tree creation
	heap.cu			: Heap sort function
=====
HOW TO COMPILE:
Simply type :
	make
=====
HOW TO RUN:
Simply type :
	./tree
	
	










