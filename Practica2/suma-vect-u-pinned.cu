#include <stdio.h>
#include <assert.h>

#define N 100000
#define tb 512	// tamaño bloque

__global__ void VecAdd(int* D)
{
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

	for (int i=ii; i<N; i+=stride){
    D[2*N + i] = D[i] + D[N + i];
  }
}

cudaError_t testCuErr(cudaError_t result)
{
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);	// si no se cumple, se aborta el programa
  }
  return result;
}

int main()
{ cudaFree(0);
  int *H,*D;
  int i, dg; int size = 3*N*sizeof(int);
  
  H = (int*)malloc(size);
  
  // reservamos espacio en la memoria global del device
  testCuErr(cudaMallocHost((void**)&D, size));
     
  // inicializamos HA y HB
  for (i=0; i<3*N; i++) {
    if(i < N){
      H[i] = -i;
    }else if((i < 2*N) && (i > N-1)){
      H[i] = 3*i;
    }
  }

  testCuErr(cudaMemcpy(D, H, size, cudaMemcpyHostToDevice));

      
  dg = (N+tb-1)/tb; if (dg>65535) dg=65535;
  // llamamos al kernel
  VecAdd <<<dg, tb>>>(D);	// N o más hilos ejecutan el kernel en paralelo
  testCuErr(cudaGetLastError());
  
  testCuErr(cudaMemcpy(H, D, size, cudaMemcpyDeviceToHost));

    
  // liberamos la memoria reservada en el device
  testCuErr(cudaFreeHost(D));
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i = 0; i < N; i++){// printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    //printf("%d = %d + %d\n",H[2*N + i],H[i],H[N + i]);
    if (H[2*N + i] != (H[i] + H[N + i])){
		printf("error en componente %d\n", i);}
  }

  free(H);
  return 0;
} 
