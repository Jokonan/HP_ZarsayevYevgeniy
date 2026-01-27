#include <mpi.h> // MPI
#include <iostream> // ввод-вывод
#include <vector> // std::vector
#include <numeric> // std::accumulate
#include <cstdlib> // rand

#define N 10000000 // размер общего массива

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // инициализация MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // число процессов

    int local_N = N / size; // размер части массива на каждом процессе
    std::vector<double> local_data(local_N); // локальный массив

    // заполняем массив случайными числами (0..1)
    for(int i=0; i<local_N; i++) local_data[i] = rand() / double(RAND_MAX);

    double local_sum = std::accumulate(local_data.begin(), local_data.end(), 0.0); // локальная сумма

    double total_sum = 0.0; // для результата
    MPI_Barrier(MPI_COMM_WORLD); // синхронизация процессов
    double t1 = MPI_Wtime(); // старт таймера

    // ----------------------
    // MPI_Reduce: суммируем все локальные суммы на процессе 0
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t2 = MPI_Wtime(); // конец таймера
    double elapsed = t2 - t1; // измеряем время операции Reduce

    if(rank == 0){
        std::cout << "=== MPI_Reduce ===\n";
        std::cout << "Total sum: " << total_sum << "\n"; // вывод суммы
        std::cout << "Elapsed time: " << elapsed << " sec\n"; // вывод времени
    }

    MPI_Barrier(MPI_COMM_WORLD); // синхронизация
    t1 = MPI_Wtime(); // старт таймера Allreduce

    // ----------------------
    // MPI_Allreduce: каждый процесс получает итоговую сумму
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    t2 = MPI_Wtime();
    elapsed = t2 - t1;

    if(rank == 0){
        std::cout << "=== MPI_Allreduce ===\n";
        std::cout << "Total sum: " << total_sum << "\n"; // вывод суммы
        std::cout << "Elapsed time: " << elapsed << " sec\n"; // вывод времени
    }

    MPI_Finalize(); // завершение MPI
    return 0;
}
