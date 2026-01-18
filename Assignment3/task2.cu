#include <iostream>
#include <cuda_runtime.h>

#define N 1000000       // размер массивов, которые будем суммировать

// Макрос для проверки ошибок CUDA
// Если вызов CUDA возвращает ошибку, выводим сообщение и останавливаем программу
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;   \
        exit(1);                                              \
    }                                                         \
} while (0)

// Ядро: поэлементное сложение двух массивов
// Каждый поток суммирует один элемент массива

__global__ void addArrays(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // вычисляем глобальный индекс для потока
    if (idx < n)                                     // проверяем, чтобы не выйти за предел массива
        C[idx] = A[idx] + B[idx];                   // суммируем элементы
}

int main() {

    // Создаем массивы на CPU

    float *h_A = new float[N]; // массив A
    float *h_B = new float[N]; // массив B
    float *h_C = new float[N]; // массив C для результата

    // Инициализируем массивы значениями
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f; // можно взять 1
        h_B[i] = 2.0f; // можно взять 2
    }


    // Выделяем память на GPU

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float))); // память для A
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float))); // память для B
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float))); // память для результата

    // Копируем данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));


    // Тестируем разные размеры блока потоков

    int block_sizes[] = {128, 256, 512};                 // три варианта размера блока
    int num_tests = sizeof(block_sizes)/sizeof(block_sizes[0]); // количество вариантов

    cudaEvent_t start, stop;    // таймеры для измерения времени выполнения
    float elapsed;              // время выполнения

    for (int t = 0; t < num_tests; ++t) {
        int block_size = block_sizes[t];                         // текущий размер блока
        int grid_size = (N + block_size - 1) / block_size;       // количество блоков для покрытия всех элементов


        // Запускаем таймер

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));                      // старт таймера


        // Запуск ядра на GPU

        addArrays<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());                           // проверка ошибок после запуска

        CUDA_CHECK(cudaEventRecord(stop));                        // останавливаем таймер
        CUDA_CHECK(cudaEventSynchronize(stop));                   // ждем завершения ядра
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));  // вычисляем время выполнения


        // Копируем результат с GPU на CPU для проверки

        CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));


        // Выводим один элемент и время выполнения
        // Это позволяет проверить корректность работы ядра

        std::cout << "Block size: " << block_size
                  << " | Sample result: " << h_C[0]
                  << " | Time: " << elapsed << " ms" << std::endl;


        // Освобождаем таймеры

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }


    // Освобождаем память на GPU и CPU

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
