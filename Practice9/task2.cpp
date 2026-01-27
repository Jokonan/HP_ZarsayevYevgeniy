#include <mpi.h>            // основной заголовок MPI для работы с MPI
#include <iostream>         // ввод-вывод в консоль
#include <vector>           // динамические массивы (std::vector)
#include <cstdlib>          // rand() и стандартные функции
#include <cmath>            // fabs() и математические функции

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);            // инициализация MPI, необходимо для работы всех MPI функций

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // определяем номер текущего процесса (rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // определяем общее количество процессов

    const int N = 8;                     // размер системы линейных уравнений (NxN)

    std::vector<double> A;                  // матрица коэффициентов (только у процесса rank 0)
    std::vector<double> b;                  // вектор правых частей (только у процесса rank 0)

    double start_time = MPI_Wtime();       // фиксируем время начала выполнения программы

    if (rank == 0) {                     // только главный процесс инициализирует данные
        A.resize(N * N);                    // выделяем память под матрицу
        b.resize(N);                      // выделяем память под вектор правых частей

        for (int i = 0; i < N; i++) {       // заполняем матрицу A
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (i == j) ? N : 1.0; // диагональ доминирующая, остальные элементы 1
            }
            b[i] = N + 1;             // заполняем вектор b
        }
    }

    int rows_per_proc = N / size;           // сколько строк матрицы будет у каждого процесса
    int remainder = N % size;             // остаток строк, если N не делится на size

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0); // сколько строк достанется текущему процессу

    // массивы для распределения матрицы A
    std::vector<int> sendcounts(size);      // сколько элементов отправлять каждому процессу
    std::vector<int> displs(size);        // смещение для каждого процесса

    // массивы для распределения вектора b
    std::vector<int> sendcounts_b(size);    // сколько элементов вектора b отправлять каждому процессу
    std::vector<int> displs_b(size);      // смещение для каждого процесса

    if (rank == 0) {                    // только главный процесс считает смещения
        int offset_A = 0;                    // смещение для матрицы A
        int offset_b = 0;                    // смещение для вектора b
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0); // количество строк для i-го процесса
            sendcounts[i] = rows * N;        // для матрицы A умножаем на N
            displs[i] = offset_A;          // смещение в массиве A
            offset_A += sendcounts[i];       // обновляем смещение

            sendcounts_b[i] = rows;          // для вектора b без умножения на N
            displs_b[i] = offset_b;          // смещение в массиве b
            offset_b += rows;           // обновляем смещение
        }
    }

    std::vector<double> local_A(local_rows * N); // локальная матрица для процесса
    std::vector<double> local_b(local_rows);   // локальный вектор для процесса

    // распределяем части матрицы A всем процессам
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // распределяем части вектора b всем процессам
    MPI_Scatterv(b.data(), sendcounts_b.data(), displs_b.data(), MPI_DOUBLE,
                 local_b.data(), local_rows, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    std::vector<double> pivot_row(N);        // опорная строка для текущего шага
    double pivot_b;             // соответствующий элемент вектора b

    int global_row = 0;            // глобальный номер первой строки текущего процесса

    // прямой ход метода Гаусса
    for (int k = 0; k < N; k++) {            // цикл по всем столбцам / ведущим строкам
        int owner = 0;                 // процесс-владелец k-й строки
        int sum = 0;                          

        // определяем, какой процесс владеет k-й строкой
        for (int p = 0; p < size; p++) {
            int rows = rows_per_proc + (p < remainder ? 1 : 0); // строки у процесса
            if (k < sum + rows) {
                owner = p;            // нашли владельца строки
                break;
            }
            sum += rows;
        }

        if (rank == owner) {               // если текущий процесс владеет строкой
            int local_k = k - sum;            // индекс строки в локальном массиве
            for (int j = 0; j < N; j++) {
                pivot_row[j] = local_A[local_k * N + j]; // копируем строку в pivot_row
            }
            pivot_b = local_b[local_k];       // копируем соответствующий элемент b
        }

        // рассылаем опорную строку и элемент b всем процессам
        MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // исключаем элементы ниже k-й строки в локальных строках
        for (int i = 0; i < local_rows; i++) { // проходим по локальным строкам
            if (global_row + i > k) {                // только строки ниже текущей
                double factor = local_A[i * N + k] / pivot_row[k]; // коэффициент для исключения
                for (int j = k; j < N; j++) {
                    local_A[i * N + j] -= factor * pivot_row[j];   // обновляем элементы матрицы
                }
                local_b[i] -= factor * pivot_b;            // обновляем элемент вектора b
            }
        }

        global_row += local_rows;        // обновляем глобальный номер строки для следующей итерации
    }

    // обратный ход метода Гаусса
    if (rank == 0) {
        std::vector<double> x(N, 0.0);       // создаём вектор решения

        for (int i = N - 1; i >= 0; i--) {   // идём снизу вверх по строкам
            x[i] = b[i];                // начинаем с правой части
            for (int j = i + 1; j < N; j++) {
                x[i] -= A[i * N + j] * x[j];  // вычитаем уже найденные значения
            }
            x[i] /= A[i * N + i];         // делим на диагональный элемент
        }

        // выводим решение в консоль
        std::cout << "Solution vector:" << std::endl;
        for (int i = 0; i < N; i++) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }

    double end_time = MPI_Wtime();            // фиксируем время окончания выполнения

    if (rank == 0) {                    // главный процесс выводит время работы
        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl;
    }

    MPI_Finalize();      // завершаем работу MPI
    return 0;
}
