#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>

using namespace std;

int heapsize;
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



