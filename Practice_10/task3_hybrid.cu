#include <iostream> // ввод-вывод
#include <cuda_runtime.h> // CUDA runtime
#include <vector> // std::vector
#include <numeric> // std::accumulate

#define N 1024*1024*16 // размер массива
#define BLOCK_SIZE 256 // размер блока
#define CHUNK_SIZE (N/4) // делим массив на 4 чанка для overlap

// ----------------------
// GPU-ядро: умножаем элементы на 2
__global__ void kernel_gpu(float* d_in, float* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    if (idx < size) d_out[idx] = d_in[idx] * 2.0f; // обработка элемента
}

int main() {
    std::vector<float> h_data(N, 1.0f); // входной массив CPU
    std::vector<float> h_out(N, 0.0f);  // массив для GPU результатов

    float *d_in, *d_out; // указатели на GPU
    size_t size = N * sizeof(float); // размер в байтах

    cudaStream_t stream1, stream2; // два потока CUDA для overlap
    cudaStreamCreate(&stream1); // создаем первый поток
    cudaStreamCreate(&stream2); // создаем второй поток

    cudaMalloc(&d_in, size); // выделяем память под входной массив
    cudaMalloc(&d_out, size); // выделяем память под выходной массив

    cudaEvent_t start, stop, t1, t2; // события CUDA для измерения времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);

    // ----------------------
    // Вариант без оптимизации
    float cpu_sum = std::accumulate(h_data.begin(), h_data.end(), 0.0f); // CPU суммирует весь массив

    // копия на GPU
    cudaEventRecord(start); // старт таймера копирования на GPU
    cudaMemcpy(d_in, h_data.data(), size, cudaMemcpyHostToDevice); // копируем на GPU
    cudaEventRecord(stop); // конец таймера
    cudaEventSynchronize(stop);
    float time_copy_to_gpu;
    cudaEventElapsedTime(&time_copy_to_gpu, start, stop); // измеряем время копирования

    // GPU вычисления
    dim3 block(BLOCK_SIZE); // размер блока
    dim3 grid((N + BLOCK_SIZE - 1)/BLOCK_SIZE); // число блоков
    cudaEventRecord(start); // старт таймера GPU
    kernel_gpu<<<grid, block>>>(d_in, d_out, N); // вызываем kernel
    cudaDeviceSynchronize(); // ждем завершения GPU
    cudaEventRecord(stop); // стоп таймера
    cudaEventSynchronize(stop);
    float time_gpu_compute;
    cudaEventElapsedTime(&time_gpu_compute, start, stop); // измеряем время GPU

    // копия обратно
    cudaEventRecord(start); // старт таймера копии обратно
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost); // копируем обратно
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_copy_to_cpu;
    cudaEventElapsedTime(&time_copy_to_cpu, start, stop); // время копии обратно

    float hybrid_sum = cpu_sum; // суммируем CPU результат
    for(int i=0; i<N; i++) hybrid_sum += h_out[i]; // добавляем GPU результат

    std::cout << "=== Before optimization ===\n"; // заголовок
    std::cout << "CPU sum: " << cpu_sum << "\n"; // вывод CPU суммы
    std::cout << "Hybrid sum: " << hybrid_sum << "\n"; // вывод гибридной суммы
    std::cout << "Time copy to GPU: " << time_copy_to_gpu << " ms\n"; // время копирования на GPU
    std::cout << "Time GPU compute: " << time_gpu_compute << " ms\n"; // время GPU
    std::cout << "Time copy to CPU: " << time_copy_to_cpu << " ms\n"; // время копирования обратно
    std::cout << "Total time: " << (time_copy_to_gpu + time_gpu_compute + time_copy_to_cpu) << " ms\n"; // общее время

    // ----------------------
    // Вариант с overlap (оптимизация)
    hybrid_sum = 0.0f; // сбрасываем сумму
    cudaMemset(d_out, 0, size); // обнуляем GPU массив

    float time_copy_to_gpu_opt=0, time_gpu_compute_opt=0, time_copy_to_cpu_opt=0; // таймеры для оптимизации

    // асинхронная обработка чанков
    for(int i=0; i<N; i+=CHUNK_SIZE){
        int chunk = std::min(CHUNK_SIZE, N-i); // размер текущего чанка
        cudaStream_t stream = (i/CHUNK_SIZE) % 2 == 0 ? stream1 : stream2; // чередуем потоки

        // копия на GPU
        cudaEventRecord(t1, stream); // старт таймера копии
        cudaMemcpyAsync(d_in+i, h_data.data()+i, chunk*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaEventRecord(t2, stream); // стоп таймера
        cudaEventSynchronize(t2);
        float tmp;
        cudaEventElapsedTime(&tmp, t1, t2); // измеряем время копии
        time_copy_to_gpu_opt += tmp; // суммируем

        // GPU вычисления
        dim3 grid_chunk((chunk + BLOCK_SIZE - 1)/BLOCK_SIZE); // число блоков для чанка
        cudaEventRecord(t1, stream); // старт таймера GPU
        kernel_gpu<<<grid_chunk, block, 0, stream>>>(d_in+i, d_out+i, chunk); // kernel
        cudaEventRecord(t2, stream); // стоп таймера
        cudaEventSynchronize(t2);
        cudaEventElapsedTime(&tmp, t1, t2);
        time_gpu_compute_opt += tmp; // суммируем

        // CPU суммирует чанк
        float cpu_chunk_sum = std::accumulate(h_data.begin()+i, h_data.begin()+i+chunk, 0.0f);
        hybrid_sum += cpu_chunk_sum; // добавляем к сумме
    }

    // копия результатов с GPU
    for(int i=0; i<N; i+=CHUNK_SIZE){
        int chunk = std::min(CHUNK_SIZE, N-i);
        cudaEventRecord(t1, stream1); // старт таймера копии
        cudaMemcpyAsync(h_out.data()+i, d_out+i, chunk*sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaEventRecord(t2, stream1); // стоп таймера
        cudaEventSynchronize(t2);
        float tmp;
        cudaEventElapsedTime(&tmp, t1, t2);
        time_copy_to_cpu_opt += tmp; // суммируем
    }

    // GPU результат добавляем к сумме
    for(int i=0; i<N; i++) hybrid_sum += h_out[i]; // финальная гибридная сумма

    std::cout << "=== After optimization (overlap CPU+GPU) ===\n";
    std::cout << "Hybrid sum: " << hybrid_sum << "\n"; // вывод суммы
    std::cout << "Time copy to GPU: " << time_copy_to_gpu_opt << " ms\n"; // время копии на GPU
    std::cout << "Time GPU compute: " << time_gpu_compute_opt << " ms\n"; // время GPU
    std::cout << "Time copy to CPU: " << time_copy_to_cpu_opt << " ms\n"; // время копии обратно
    std::cout << "Total time: " << (time_copy_to_gpu_opt + time_gpu_compute_opt + time_copy_to_cpu_opt) << " ms\n"; // итоговое время

    // ----------------------
    cudaFree(d_in); // освобождаем память
    cudaFree(d_out);
    cudaStreamDestroy(stream1); // уничтожаем потоки
    cudaStreamDestroy(stream2);

    return 0;
}
