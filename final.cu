// scan_benchmark.cu
// Полный бенчмарк для сравнения префиксных сумм (inclusive scan)
// Реализации: последовательная CPU, параллельная CPU (OpenMP) и GPU (CUDA)

#include <cuda_runtime.h>          // Основные функции и типы CUDA
#include <iostream>                // Работа с консольным вводом/выводом
#include <vector>                  // Контейнер динамических массивов
#include <random>                  // Генерация случайных чисел
#include <chrono>                  // Измерение времени
#include <iomanip>                 // Форматированный вывод таблиц
#include <cassert>                 // Макросы для проверок
#include <omp.h>                   // Поддержка OpenMP

// ============================================================
// Макрос проверки ошибок CUDA
// ============================================================
// Используется после каждого CUDA-вызова
// Если произошла ошибка — программа завершается
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do {                                         \
    cudaError_t _e = (call);                                           /* Выполняем CUDA-функцию */ \
    if (_e != cudaSuccess) {                                           /* Проверяем код ошибки */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e)          /* Печатаем описание */ \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";    /* Указываем место */ \
        std::exit(1);                                                  /* Прерываем выполнение */ \
    }                                                                  \
} while(0)
#endif

// ============================================================
// 1) Последовательный inclusive scan на CPU
// ============================================================
// Каждый элемент выходного массива — это сумма всех предыдущих
// элементов входного массива, включая текущий
void scan_inclusive_seq(const int* in, int* out, size_t n) {
    if (n == 0) return;             // Если размер 0 — ничего не делаем
    long long acc = 0;              // Переменная для накопления суммы
    for (size_t i = 0; i < n; i++) { // Проходим по всему массиву
        acc += in[i];               // Добавляем текущий элемент
        out[i] = (int)acc;          // Записываем накопленную сумму
    }
}

// ============================================================
// 2) Inclusive scan на CPU с использованием OpenMP
// ============================================================
// Алгоритм состоит из двух логических фаз:
// 1) Каждый поток считает свой кусок массива
// 2) К каждому куску добавляется сумма предыдущих кусков
void scan_inclusive_omp(const int* in, int* out, size_t n) {
    if (n == 0) return;             // Проверка на пустой массив

    int num_threads = 1;            // Переменная для числа потоков
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads(); // Определяем число потоков
    }

    std::vector<long long> chunk_sums(num_threads, 0); // Суммы кусков

#pragma omp parallel
    {
        int tid = omp_get_thread_num(); // Номер текущего потока
        int T   = omp_get_num_threads();// Общее число потоков

        size_t chunk = (n + T - 1) / T; // Размер куска на поток
        size_t start = (size_t)tid * chunk; // Начальный индекс
        size_t end   = std::min(n, start + chunk); // Конечный индекс

        // ---------- Фаза 1: локальный scan ----------
        long long acc = 0;          // Локальная сумма потока
        for (size_t i = start; i < end; i++) {
            acc += in[i];           // Складываем элементы
            out[i] = (int)acc;      // Пишем локальный результат
        }
        chunk_sums[tid] = acc;      // Сохраняем сумму всего куска

#pragma omp barrier                // Ждём, пока все потоки закончат фазу 1

        // ---------- Фаза 2: вычисление смещения ----------
        long long offset = 0;       // Смещение для текущего потока
        for (int k = 0; k < tid; k++) offset += chunk_sums[k]; // Сумма предыдущих

        // ---------- Фаза 3: добавление смещения ----------
        for (size_t i = start; i < end; i++) {
            out[i] = (int)((long long)out[i] + offset); // Коррекция результата
        }
    }
}

// ============================================================
// 3) CUDA-реализация inclusive scan (двухфазная)
// ============================================================
// Общая идея:
// - scan внутри блоков
// - рекурсивный scan сумм блоков
// - добавление смещений к каждому блоку

// ------------------------------------------------------------
// Scan внутри одного warp (32 потока)
// ------------------------------------------------------------
static __device__ __forceinline__ int warpInclusiveScan(int val) {
    // Последовательно складываем значения от соседних потоков
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset); // Берём значение слева
        if ((threadIdx.x & 31) >= offset) val += n;      // Прибавляем, если можно
    }
    return val;                      // Возвращаем результат
}

// ------------------------------------------------------------
// Kernel: scan внутри одного CUDA-блока
// ------------------------------------------------------------
__global__ void block_scan_inclusive(const int* in, int* out, int* blockSums, int n) {
    extern __shared__ int smem[];    // Общая память для сумм warp
    int tid = threadIdx.x;           // Индекс потока в блоке
    int gid = blockIdx.x * blockDim.x + tid; // Глобальный индекс элемента

    int x = (gid < n) ? in[gid] : 0; // Загружаем данные или 0

    // ---------- Шаг 1: scan внутри warp ----------
    int val = warpInclusiveScan(x);

    int lane = tid & 31;             // Позиция в warp
    int warp = tid >> 5;             // Номер warp в блоке

    // ---------- Шаг 2: запись сумм warp ----------
    if (lane == 31) smem[warp] = val; // Последний поток warp пишет сумму
    __syncthreads();                 // Синхронизация потоков

    // ---------- Шаг 3: scan сумм warp ----------
    if (warp == 0) {                 // Только первый warp
        int wval = (tid < (blockDim.x + 31) / 32) ? smem[lane] : 0;
        wval = warpInclusiveScan(wval); // Scan сумм warp
        smem[lane] = wval;              // Сохраняем результат
    }
    __syncthreads();                 // Синхронизация

    // ---------- Шаг 4: добавление смещения warp ----------
    if (warp > 0) {
        val += smem[warp - 1];       // Добавляем сумму предыдущих warp
    }

    if (gid < n) out[gid] = val;     // Записываем результат блока

    // ---------- Шаг 5: сохранение суммы блока ----------
    if (blockSums) {
        __syncthreads();             // Ждём, пока out заполнится
        if (tid == 0) {
            int last = min((blockIdx.x + 1) * blockDim.x, n) - 1;
            blockSums[blockIdx.x] = out[last]; // Сумма всего блока
        }
    }
}

// ------------------------------------------------------------
// Kernel: добавление смещений блоков
// ------------------------------------------------------------
__global__ void add_block_offsets(int* data, const int* scannedBlockSums, int n) {
    int tid = threadIdx.x;           // Индекс потока
    int bid = blockIdx.x;            // Индекс блока
    int gid = bid * blockDim.x + tid;// Глобальный индекс элемента
    if (gid >= n) return;            // Проверка границ
    if (bid == 0) return;            // Первый блок не смещаем
    int offset = scannedBlockSums[bid - 1]; // Смещение для блока
    data[gid] += offset;             // Добавляем смещение
}

// ------------------------------------------------------------
// Рекурсивный scan массивов на GPU
// ------------------------------------------------------------
void gpu_scan_recursive(const int* d_in, int* d_out, int n, int blockSize) {
    int numBlocks = (n + blockSize - 1) / blockSize; // Количество блоков

    int* d_blockSums = nullptr;      // Массив сумм блоков
    if (numBlocks > 1) {
        CHECK_CUDA(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));
    }

    size_t shmem = ((blockSize + 31) / 32) * sizeof(int); // Shared memory
    block_scan_inclusive<<<numBlocks, blockSize, shmem>>>(d_in, d_out, d_blockSums, n);
    CHECK_CUDA(cudaGetLastError());  // Проверяем запуск kernel

    if (numBlocks > 1) {
        int* d_scannedBlockSums = nullptr; // Scan сумм блоков
        CHECK_CUDA(cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(int)));

        gpu_scan_recursive(d_blockSums, d_scannedBlockSums, numBlocks, blockSize);

        add_block_offsets<<<numBlocks, blockSize>>>(d_out, d_scannedBlockSums, n);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaFree(d_scannedBlockSums)); // Освобождение памяти
        CHECK_CUDA(cudaFree(d_blockSums));        // Освобождение памяти
    }
}

// ============================================================
// Обёртка для CUDA scan + замеры времени
// ============================================================
struct GpuTiming {
    float ms_kernels = 0.0f;         // Время только kernel
    float ms_total   = 0.0f;         // Общее время (копии + kernel)
};

GpuTiming scan_inclusive_cuda(const std::vector<int>& h_in, std::vector<int>& h_out, int blockSize) {
    const int n = (int)h_in.size();  // Размер входного массива
    h_out.assign(n, 0);              // Инициализация выходного массива

    int *d_in=nullptr, *d_out=nullptr; // Указатели на GPU-память
    CHECK_CUDA(cudaMalloc(&d_in,  n * sizeof(int))); // Выделяем память
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(int))); // Выделяем память

    cudaEvent_t e0,e1,e2,e3;         // CUDA-события
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2));
    CHECK_CUDA(cudaEventCreate(&e3));

    CHECK_CUDA(cudaEventRecord(e0)); // Старт общего времени
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), n*sizeof(int), cudaMemcpyHostToDevice)); // H2D
    CHECK_CUDA(cudaEventRecord(e1)); // Конец копирования

    CHECK_CUDA(cudaEventRecord(e2)); // Старт kernel
    gpu_scan_recursive(d_in, d_out, n, blockSize);   // Запуск scan
    CHECK_CUDA(cudaEventRecord(e3)); // Конец kernel
    CHECK_CUDA(cudaEventSynchronize(e3));             // Ждём завершения

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n*sizeof(int), cudaMemcpyDeviceToHost)); // D2H

    float ms_h2d=0, ms_k=0, ms_total=0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, e0, e1)); // Время копирования
    CHECK_CUDA(cudaEventElapsedTime(&ms_k,   e2, e3)); // Время kernel

    cudaEvent_t e4;
    CHECK_CUDA(cudaEventCreate(&e4));
    CHECK_CUDA(cudaEventRecord(e4));
    CHECK_CUDA(cudaEventSynchronize(e4));
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, e0, e4)); // Полное время
    CHECK_CUDA(cudaEventDestroy(e4));

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaEventDestroy(e2));
    CHECK_CUDA(cudaEventDestroy(e3));

    CHECK_CUDA(cudaFree(d_in));      // Освобождаем GPU-память
    CHECK_CUDA(cudaFree(d_out));     // Освобождаем GPU-память

    return {ms_k, ms_total};         // Возвращаем результаты
}

// ============================================================
// Вспомогательные функции
// ============================================================
double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

bool check_equal(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false; // Проверка размеров
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return false;     // Проверка значений
    }
    return true;
}

// ============================================================
// Главная функция программы
// ============================================================
int main() {
    std::vector<size_t> sizes = {   // Набор размеров для тестов
        10000,
        100000,
        1000000,
        10000000,
        20000000,
        50000000
    };

    std::mt19937 rng(12345);         // Инициализация генератора
    std::uniform_int_distribution<int> dist(0, 5); // Диапазон чисел

    int blockSize = 256;             // Размер блока CUDA

    std::cout << "Inclusive prefix-sum (scan) benchmark\n"; // Заголовок
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n"; // Потоки CPU
    std::cout << "CUDA blockSize: " << blockSize << "\n\n"; // Размер блока

    // Заголовок таблицы результатов
    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(12) << "Seq(ms)"
              << std::setw(12) << "OMP(ms)"
              << std::setw(14) << "GPUkern(ms)"
              << std::setw(14) << "GPUtot(ms)"
              << std::setw(10) << "OMP ok"
              << std::setw(10) << "GPU ok"
              << std::setw(12) << "Spd OMP"
              << std::setw(12) << "Spd GPU(tot)"
              << "\n";

    std::cout << std::string(110, '-') << "\n"; // Разделительная линия

    for (size_t n : sizes) {         // Цикл по всем размерам
        std::vector<int> in(n);      // Входной массив
        for (size_t i = 0; i < n; i++) in[i] = dist(rng); // Заполняем данными

        std::vector<int> out_seq(n), out_omp(n), out_gpu(n); // Результаты

        double t0 = now_ms();
        scan_inclusive_seq(in.data(), out_seq.data(), n); // Последовательный scan
        double t1 = now_ms();
        double ms_seq = t1 - t0;     // Время CPU seq

        double t2 = now_ms();
        scan_inclusive_omp(in.data(), out_omp.data(), n); // OpenMP scan
        double t3 = now_ms();
        double ms_omp = t3 - t2;     // Время CPU OMP

        bool ok_omp = check_equal(out_seq, out_omp); // Проверка корректности OMP

        GpuTiming gt = scan_inclusive_cuda(in, out_gpu, blockSize); // GPU scan
        bool ok_gpu = check_equal(out_seq, out_gpu); // Проверка корректности GPU

        double spd_omp = ms_seq / ms_omp;            // Ускорение OMP
        double spd_gpu = ms_seq / gt.ms_total;       // Ускорение GPU

        // Вывод строки таблицы
        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(12) << std::fixed << std::setprecision(3) << ms_seq
                  << std::setw(12) << ms_omp
                  << std::setw(14) << gt.ms_kernels
                  << std::setw(14) << gt.ms_total
                  << std::setw(10) << (ok_omp ? "OK" : "FAIL")
                  << std::setw(10) << (ok_gpu ? "OK" : "FAIL")
                  << std::setw(12) << spd_omp
                  << std::setw(12) << spd_gpu
                  << "\n";

        if (!ok_omp || !ok_gpu) {    // Если ошибка — сразу выходим
            std::cerr << "\nCorrectness check failed at N=" << n << "\n";
            return 2;
        }
    }

    // Пояснения к результатам
    std::cout << "\nNotes:\n"
              << "- GPUkern(ms): время выполнения только CUDA kernel\n"
              << "- GPUtot(ms): полное время (копирование + kernel)\n";

    return 0;
}
