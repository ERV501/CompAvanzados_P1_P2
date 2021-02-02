#include <stdio.h>

#define N 500

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int i = threadIdx.x;
    DC[i] = DA[i] + DB[i];
}

int main()
{ 
  cudaFree(0);
  
  int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;
  int i; int size = N*sizeof(int);
  cudaError_t aM,bM,cM, aN, bN, cN, e_kernel; //Guardar errores

  // reservamos espacio en la memoria global del device

  aM = cudaMalloc((void**)&DA, size);
  printf("%s \n",cudaGetErrorString(aM));

  bM = cudaMalloc((void**)&DB, size);
  printf("%s \n",cudaGetErrorString(bM));

  cM = cudaMalloc((void**)&DC, size);
  printf("%s \n",cudaGetErrorString(cM));
  
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  aN = cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  printf("%s \n",cudaGetErrorString(aN));

  bN = cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  printf("%s \n",cudaGetErrorString(bN));

  // llamamos al kernel (1 bloque de N hilos)
  VecAdd <<<1, N>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo
  e_kernel = cudaGetLastError(); //Cojer ultimo error, ya que el kernel no devuelve ningun error_t
  printf("%s \n",cudaGetErrorString(e_kernel)); //Imprimir ultimo error

  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  cN = cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  printf("%s \n",cudaGetErrorString(cN));

  // liberamos la memoria reservada en el device
  cudaFree(DA); cudaFree(DB); cudaFree(DC);  
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  // esta comprobación debe quitarse una vez que el programa es correcto (p. ej., para medir el tiempo de ejecución)
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}
  
  return 0;
} 
