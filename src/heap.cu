#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>

using namespace std;

int heapsize;
void MaxHeapify4(int *data, double *data1, double *data2, double *data3, int *data4, int root)
{
        int l=2*root+1;
        int r=2*root+2;
        int largest;
      	double temp;
	int tempi;
        if( l<heapsize & r<heapsize ){
                if( data[l]>data[root] || data[r]>data[root] ){
                        if( data[l]>data[r] ){ largest = l; }
                        else{ largest = r; }
                }else{
                        largest = root;
                }
        }else if( l<heapsize ){
                if( data[l] > data[root] ){ largest = l;}
                else{ largest = root; }
        }else if( r<heapsize ){
                if( data[r] > data[root] ){ largest = r;}
                else{ largest = root; }
        }else{
                largest = root;
        }
        if( largest != root ){
                tempi = data[root];
                data[root] = data[largest];
                data[largest] = tempi;
                temp = data1[root];
                data1[root] = data1[largest];
                data1[largest] = temp;
		temp = data2[root];
                data2[root] = data2[largest];
                data2[largest] = temp;
		temp = data3[root];
                data3[root] = data3[largest];
                data3[largest] = temp;
		tempi = data4[root];
                data4[root] = data4[largest];
                data4[largest] = tempi;
                MaxHeapify4(data,data1,data2,data3,data4,largest);
        }
}

void BuildMaxHeapify4(int *data,double *data1,double *data2,double *data3,int *data4,int n)
{
        heapsize = n;
        for( int i=n/2;i>=0;i-- ){
                MaxHeapify4(data,data1,data2,data3,data4,i);
        }
}

void HeapSort4(int *data, double *data1, double *data2, double *data3, int *data4, int n){
        BuildMaxHeapify4(data,data1,data2,data3,data4,n);
//        cout<<"Finish max heapify,heapsize="<<heapsize<<endl;
	int tmpi;
	double tmp;
        for( int r=n-1;r>=1;r-- ){
                tmpi = data[r];
                data[r] = data[0];
                data[0] = tmpi;
                tmp = data1[r];
                data1[r] = data1[0];
                data1[0] = tmp;
		tmp = data2[r];
                data2[r] = data2[0];
                data2[0] = tmp;
		tmp = data3[r];
                data3[r] = data3[0];
                data3[0] = tmp;
		tmpi = data4[r];
		data4[r] = data4[0];
		data4[0] = tmpi;
                heapsize--;
                MaxHeapify4(data,data1,data2,data3,data4,0);
        }
}

void MaxHeapify(int *data, int *data_f, int root)
{
        int l=2*root+1;
        int r=2*root+2;
        int largest, temp;
        if( l<heapsize & r<heapsize ){
                if( data[l]>data[root] || data[r]>data[root] ){
                        if( data[l]>data[r] ){ largest = l; }
                        else{ largest = r; }
                }else{
                        largest = root;
                }
        }else if( l<heapsize ){
                if( data[l] > data[root] ){ largest = l;}
                else{ largest = root; }
        }else if( r<heapsize ){
                if( data[r] > data[root] ){ largest = r;}
                else{ largest = root; }
        }else{
                largest = root;
        }
        if( largest != root ){
                temp = data[root];
                data[root] = data[largest];
                data[largest] = temp;
                temp = data_f[root];
                data_f[root] = data_f[largest];
                data_f[largest] = temp;
                MaxHeapify(data,data_f,largest);
        }
}

void BuildMaxHeapify(int *data,int *data_f,int n)
{
        heapsize = n;
        for( int i=n/2;i>=0;i-- ){
                MaxHeapify(data,data_f,i);
        }
}

void HeapSort(int *data, int *data_f, int n){
        BuildMaxHeapify(data,data_f,n);
//        cout<<"Finish max heapify,heapsize="<<heapsize<<endl;
        int tmp = 0;
        for( int r=n-1;r>=1;r-- ){
                tmp = data[r];
                data[r] = data[0];
                data[0] = tmp;
                tmp = data_f[r];
                data_f[r] = data_f[0];
                data_f[0] = tmp;
                heapsize--;
                MaxHeapify(data,data_f,0);
        }
}

void MinHeapify(int *data, int *data_f, int root)
{
        int l=2*root+1;
        int r=2*root+2;
        int largest, temp;
        if( l<heapsize && r<heapsize ){
                if( data[l]<data[root] || data[r]<data[root] ){
                        if( data[l]<data[r] ){ largest = l; }
                        else{ largest = r; }
                }else{
                        largest = root;
                }
        }else if( l<heapsize ){
                if( data[l] < data[root] ){ largest = l;}
                else{ largest = root; }
        }else if( r<heapsize ){
                if( data[r] < data[root] ){ largest = r;}
                else{ largest = root; }
        }else{
                largest = root;
        }
        if( largest != root ){
                temp = data[root];
                data[root] = data[largest];
                data[largest] = temp;
                temp = data_f[root];
                data_f[root] = data_f[largest];
                data_f[largest] = temp;
                MinHeapify(data,data_f,largest);
        }
}

void BuildMinHeapify(int *data,int *data_f,int n)
{
        heapsize = n;
        for( int i=n/2;i>=0;i-- ){
                MinHeapify(data,data_f,i);
        }
}

