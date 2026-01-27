#include <iostream>                 // стандартный ввод-вывод
#include <vector>                   // контейнер vector
#include <chrono>                   // измерение времени
#include <omp.h>                    // OpenMP для CPU
#include <cuda_runtime.h>           // CUDA runtime API


// CUDA ядра


// CUDA ядро: умножает элементы массива на 2
__global__ void gpu_mul2(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    if (idx < n)                            // проверка выхода за границы
        data[idx] *= 2.0f;                          // умножаем элемент
}


// Один тест для размера N

void run_test(int N)
{
    std::cout << "\n==============================\n"; 
    std::cout << "Array size N = " << N << std::endl; // вывод размера массива

    int half = N / 2;                       // половина массива
    int threads = 256;                           // потоков в блоке CUDA
    int blocksN = (N + threads - 1) / threads;         // блоки для всего массива
    int blocksHalf = (half + threads - 1) / threads;   // блоки для половины

    // Исходный массив
    std::vector<float> data(N, 1.0f);                  // массив, заполненный 1.0

    
    // ЗАДАНИЕ 1: CPU + OpenMP (умножение на 2)
    
    auto cpu_data = data;                              // копия массива для CPU

    auto cpu_start = std::chrono::high_resolution_clock::now(); // старт таймера CPU

    #pragma omp parallel for                    // параллельный цикл OpenMP
    for (int i = 0; i < N; i++)                 // проходим по массиву
        cpu_data[i] *= 2.0f;                   // умножаем элемент

    auto cpu_end = std::chrono::high_resolution_clock::now(); // конец таймера CPU

    double cpu_time =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // время CPU

    // ЗАДАНИЕ 2: GPU + CUDA (умножение на 2)

    auto gpu_data = data;                              // копия массива для GPU

    float* d_data;                           // указатель на память GPU
    cudaMalloc(&d_data, N * sizeof(float));            // выделяем память на GPU
    cudaMemcpy(d_data, gpu_data.data(),
               N * sizeof(float), cudaMemcpyHostToDevice); // копируем данные CPU → GPU

    cudaEvent_t g_start, g_stop;                       // CUDA события для времени
    cudaEventCreate(&g_start);                         // создаем событие старта
    cudaEventCreate(&g_stop);                          // создаем событие конца

    cudaEventRecord(g_start);                          // старт замера GPU времени
    gpu_mul2<<<blocksN, threads>>>(d_data, N);         // запуск CUDA ядра
    cudaDeviceSynchronize();                           // ждем завершения GPU
    cudaEventRecord(g_stop);                           // конец замера
    cudaEventSynchronize(g_stop);                      // синхронизация события

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, g_start, g_stop);  // вычисляем время GPU

    cudaMemcpy(gpu_data.data(), d_data,
               N * sizeof(float), cudaMemcpyDeviceToHost); // копируем результат GPU → CPU

    cudaFree(d_data);                     // освобождаем память GPU

    
    // ЗАДАНИЕ 3: ГИБРИД (Одна операция)
    
    auto hybrid_same = data;                    // массив для гибридной обработки

    cudaMalloc(&d_data, half * sizeof(float));         // память GPU под половину массива
    cudaMemcpy(d_data, hybrid_same.data() + half,
               half * sizeof(float), cudaMemcpyHostToDevice); // копируем вторую половину

    auto hybrid_same_start = std::chrono::high_resolution_clock::now(); // старт гибридного таймера

    // CPU часть — первая половина
    #pragma omp parallel for                            // параллельный CPU цикл
    for (int i = 0; i < half; i++)                     // первая половина массива
        hybrid_same[i] *= 2.0f;                        // умножаем элементы

    // GPU часть — вторая половина
    gpu_mul2<<<blocksHalf, threads>>>(d_data, half);   // запуск ядра для половины
    cudaDeviceSynchronize();                           // ждем GPU

    cudaMemcpy(hybrid_same.data() + half, d_data,
               half * sizeof(float), cudaMemcpyDeviceToHost); // возвращаем данные с GPU

    auto hybrid_same_end = std::chrono::high_resolution_clock::now(); // конец таймера

    double hybrid_same_time =
        std::chrono::duration<double, std::milli>(
            hybrid_same_end - hybrid_same_start).count(); // время гибридной обработки

    cudaFree(d_data);                                  // освобождаем память GPU

    // ДОП. ЗАДАНИЕ 1: ГИБРИД (CPU +1, GPU *2)

    auto hybrid_diff = data;                           // массив для второй гибридной версии

    cudaMalloc(&d_data, half * sizeof(float));         // память GPU под вторую половину
    cudaMemcpy(d_data, hybrid_diff.data() + half,
               half * sizeof(float), cudaMemcpyHostToDevice); // копируем данные на GPU

    auto hybrid_diff_start = std::chrono::high_resolution_clock::now(); // старт таймера

    // CPU часть — сложение
    #pragma omp parallel for                         // параллельный CPU цикл
    for (int i = 0; i < half; i++)                     // первая половина массива
        hybrid_diff[i] += 1.0f;                       // прибавляем 1

    // GPU часть — умножение
    gpu_mul2<<<blocksHalf, threads>>>(d_data, half);   // GPU умножает вторую половину
    cudaDeviceSynchronize();                        // ждем завершения GPU

    cudaMemcpy(hybrid_diff.data() + half, d_data,
               half * sizeof(float), cudaMemcpyDeviceToHost); // копируем результат

    auto hybrid_diff_end = std::chrono::high_resolution_clock::now(); // конец таймера

    double hybrid_diff_time =
        std::chrono::duration<double, std::milli>(
            hybrid_diff_end - hybrid_diff_start).count(); // время второй гибридной версии

    cudaFree(d_data);                               // освобождаем память GPU


    // Вывод результатов---------------------------------------------

    std::cout << "CPU (OpenMP):            " << cpu_time << " ms\n";      // время CPU
    std::cout << "GPU (CUDA):              " << gpu_time << " ms\n";      // время GPU
    std::cout << "Hybrid SAME (одна операция):       " << hybrid_same_time << " ms\n"; // гибрид 1
    std::cout << "Hybrid DIFFERENT (разные операции):  " << hybrid_diff_time << " ms\n"; // гибрид 2
}

int main()
{
    run_test(100000);                                  // тест с маленьким массивом
    run_test(500000);                                  // средний размер
    run_test(1000000);                                 // большой массив
    run_test(5000000);                                 // очень большой массив

    return 0;
}
