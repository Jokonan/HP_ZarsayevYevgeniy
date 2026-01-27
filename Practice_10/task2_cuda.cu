#include <iostream>
#include <cuda_runtime.h>

#define N 1024*1024*32  // размер массива
#define BLOCK_SIZE 256  // количество потоков в блоке

__global__ void kernel_coalesced(float* d_in, float* d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    if (idx < N) d_out[idx] = d_in[idx] * 2.0f; // обработка элемента
}

__global__ void kernel_noncoalesced(float* d_in, float* d_out) {
    int idx = threadIdx.x * gridDim.x + blockIdx.x; // плохой паттерн доступа
    if (idx < N) d_out[idx] = d_in[idx] * 2.0f; 
}

__global__ void kernel_shared_memory(float* d_in, float* d_out) {
    __shared__ float sdata[BLOCK_SIZE]; // shared memory блок
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    int tid = threadIdx.x; // индекс в блоке
    if (idx < N) sdata[tid] = d_in[idx]; // читаем в shared memory
    __syncthreads(); // синхронизируем потоки
    if (idx < N) d_out[idx] = sdata[tid] * 2.0f; // записываем результат
}

__global__ void kernel_thread_optimization(float* d_in, float* d_out) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // 2 элемента на поток
    if (idx < N) d_out[idx] = d_in[idx] * 2.0f; // первый элемент
    if (idx + blockDim.x < N) d_out[idx + blockDim.x] = d_in[idx + blockDim.x] * 2.0f; // второй
}

int main() {
    float *h_in, *h_out; // массивы на CPU
    float *d_in, *d_out; // массивы на GPU
    size_t size = N * sizeof(float); // размер в байтах

    h_in = new float[N]; // выделяем память CPU
    h_out = new float[N]; // выделяем память CPU

    for (int i = 0; i < N; i++) h_in[i] = 1.0f; // инициализируем данные

    cudaMalloc(&d_in, size); // память GPU
    cudaMalloc(&d_out, size); // память GPU

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice); // копируем на GPU

    dim3 block(BLOCK_SIZE); // число потоков на блок
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE); // число блоков

    cudaEvent_t start, stop; // таймеры

    cudaEventCreate(&start); // событие старта
    cudaEventCreate(&stop); // событие конца

    // ----------------------
    // коалесцированный доступ
    cudaEventRecord(start); // старт
    kernel_coalesced<<<grid, block>>>(d_in, d_out); // запуск ядра
    cudaEventRecord(stop); // стоп
    cudaEventSynchronize(stop); // ждем
    float time_coalesced;
    cudaEventElapsedTime(&time_coalesced, start, stop); // время

    // ----------------------
    // некоалесцированный доступ
    cudaEventRecord(start);
    kernel_noncoalesced<<<grid, block>>>(d_in, d_out); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_noncoalesced;
    cudaEventElapsedTime(&time_noncoalesced, start, stop); 

    // ----------------------
    // использование shared memory
    cudaEventRecord(start);
    kernel_shared_memory<<<grid, block>>>(d_in, d_out); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop); 

    // ----------------------
    // оптимизация потоков (несколько элементов)
    dim3 grid_thread_opt((N + BLOCK_SIZE*2 - 1) / (BLOCK_SIZE*2)); // новая сетка
    cudaEventRecord(start);
    kernel_thread_optimization<<<grid_thread_opt, block>>>(d_in, d_out); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_thread_opt;
    cudaEventElapsedTime(&time_thread_opt, start, stop); 

    // ----------------------
    // вывод результатов
    std::cout << "Coalesced access time: " << time_coalesced << " ms\n"; 
    std::cout << "Non-coalesced access time: " << time_noncoalesced << " ms\n"; 
    std::cout << "Shared memory time: " << time_shared << " ms\n"; 
    std::cout << "Thread optimization time: " << time_thread_opt << " ms\n"; 

    cudaFree(d_in); // освобождаем GPU
    cudaFree(d_out); 
    delete[] h_in; // освобождаем CPU
    delete[] h_out; 

    return 0;
}
