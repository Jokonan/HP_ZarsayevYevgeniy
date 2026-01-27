#include <mpi.h>          // основной заголовок MPI
#include <iostream>      // ввод и вывод в консоль
#include <vector>        // динамические массивы
#include <cstdlib>       // rand(), RAND_MAX
#include <cmath>         // sqrt()

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);              // инициализация MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // общее количество процессов

    const int N = 1000000;               // размер исходного массива

    std::vector<double> data;            // массив данных
    std::vector<int> sendcounts(size);   // сколько элементов отправлять каждому процессу
    std::vector<int> displs(size);        // смещения для Scatterv

    double start_time = MPI_Wtime();      // начало измерения времени

    if (rank == 0) {             // выполняется только на главном процессе
        data.resize(N);             // выделяем память под массив
        for (int i = 0; i < N; i++) {     // заполняем массив случайными числами
            data[i] = static_cast<double>(rand()) / RAND_MAX;
        }

        int base = N / size;         // минимальное количество элементов на процесс
        int remainder = N % size;         // остаток от деления

        int offset = 0;              // текущее смещение в массиве
        for (int i = 0; i < size; i++) {  // для каждого процесса
            sendcounts[i] = base + (i < remainder ? 1 : 0); // учитываем остаток
            displs[i] = offset;         // запоминаем смещение
            offset += sendcounts[i];      // сдвигаем указатель
        }
    }

    int local_n;                     // сколько элементов получит процесс

    MPI_Scatter(sendcounts.data(), 1, MPI_INT, // отправляем размеры кусков
                &local_n, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    std::vector<double> local_data(local_n); // локальный массив процесса

    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, // распределяем массив
                 local_data.data(), local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double local_sum = 0.0;             // сумма элементов локального массива
    double local_sq_sum = 0.0;            // сумма квадратов элементов

    for (int i = 0; i < local_n; i++) {   // перебираем локальные данные
        local_sum += local_data[i];     // считаем сумму
        local_sq_sum += local_data[i] * local_data[i]; // считаем сумму квадратов
    }

    double global_sum = 0.0;            // общая сумма
    double global_sq_sum = 0.0;           // общая сумма квадратов

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // собираем суммы
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // собираем квадраты

    double end_time = MPI_Wtime();        // конец измерения времени

    if (rank == 0) {                  // только главный процесс выводит результат
        double mean = global_sum / N;    // вычисляем среднее значение
        double variance = (global_sq_sum / N) - mean * mean; // дисперсия
        double stddev = std::sqrt(variance); // стандартное отклонение

        std::cout << "Mean: " << mean << std::endl;
        std::cout << "Standard deviation: " << stddev << std::endl;
        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl;
    }

    MPI_Finalize();               // завершение работы MPI
    return 0;
}
