#include <mpi.h>             // основной заголовок MPI
#include <iostream>          // ввод-вывод в консоль
#include <vector>            // динамические массивы std::vector
#include <cstdlib>           // rand() и стандартные функции
#include <ctime>             // для srand(time(0))
#include <limits>            // для INF

const double INF = 1e9;       // обозначаем "бесконечность" для отсутствующих рёбер

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);                // инициализация MPI, необходимо для всех MPI функций

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // получаем номер текущего процесса (rank)
    MPI_Comm_size(MPI_COMM_WORLD, &size);     // получаем общее количество процессов

    int N = 8;                         // размер графа NxN
    std::vector<double> adjacency;            // матрица смежности

    double start_time = MPI_Wtime();          // фиксируем время начала программы

    if (rank == 0) {                      // только главный процесс создаёт граф
        adjacency.resize(N * N);              // выделяем память под матрицу смежности
        srand(time(0));                     // инициализация генератора случайных чисел

        // создаём случайный граф
        for (int i = 0; i < N; i++) {       // цикл по строкам
            for (int j = 0; j < N; j++) {     // цикл по столбцам
                if (i == j) adjacency[i * N + j] = 0;       // расстояние до себя = 0
                else {
                    int r = rand() % 10;                        
                    adjacency[i * N + j] = (r < 7) ? r + 1 : INF; // случайный вес или INF
                }
            }
        }

        // выводим исходную матрицу
        std::cout << "Initial adjacency matrix:" << std::endl;
        for (int i = 0; i < N; i++) {       // вывод строк
            for (int j = 0; j < N; j++) {     // вывод столбцов
                if (adjacency[i * N + j] >= INF) std::cout << "INF "; // INF вместо больших чисел
                else std::cout << adjacency[i * N + j] << " ";   // обычный вес
            }
            std::cout << std::endl;            // переход на новую строку
        }
        std::cout << std::endl;               // пустая строка для читаемости
    }

    // распределение строк между процессами
    int rows_per_proc = N / size;            // базовое количество строк на процесс
    int remainder = N % size;             // остаток строк
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0); // строки для текущего процесса

    // создаём массивы для Scatterv
    std::vector<int> sendcounts(size);       // сколько элементов отправить каждому процессу
    std::vector<int> displs(size);         // смещение для каждого процесса
    int offset = 0;                          

    for (int i = 0; i < size; i++) {        // вычисляем sendcounts и displs для каждого процесса
        int rows = rows_per_proc + (i < remainder ? 1 : 0); // строки для i-го процесса
        sendcounts[i] = rows * N;           // количество элементов в матрице
        displs[i] = offset;              // смещение для Scatterv
        offset += sendcounts[i];            // сдвиг для следующего процесса
    }

    std::vector<double> local_matrix(local_rows * N); // локальная часть матрицы для процесса

    // распределяем строки графа между процессами
    MPI_Scatterv(adjacency.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_matrix.data(), local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    std::vector<double> global_matrix(N * N);       // буфер для обмена всеми строками между процессами

    // главный цикл алгоритма Флойда-Уоршелла
    for (int k = 0; k < N; k++) {                // проход по каждой вершине k
        // определяем процесс-владельца строки k
        int owner = 0;                                
        int sum = 0;
        for (int p = 0; p < size; p++) {             
            int rows = rows_per_proc + (p < remainder ? 1 : 0); // строки процесса
            if (k < sum + rows) {                     
                owner = p;                // нашли процесс-владельца строки k
                break;
            }
            sum += rows;                               
        }

        std::vector<double> pivot_row(N);             // строка k, которая будет рассылается всем процессам

        if (rank == owner) {                      // владелец строки копирует её в pivot_row
            int local_k = k - sum;                    // локальный индекс строки
            for (int j = 0; j < N; j++) {            
                pivot_row[j] = local_matrix[local_k * N + j]; // копируем строку
            }
        }

        // рассылаем строку k всем процессам
        MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // обновляем локальные строки
        for (int i = 0; i < local_rows; i++) {       // проходим по локальным строкам
            for (int j = 0; j < N; j++) {           // проходим по столбцам
                // если можно сократить путь через вершину k
                if (local_matrix[i * N + k] + pivot_row[j] < local_matrix[i * N + j]) {
                    local_matrix[i * N + j] = local_matrix[i * N + k] + pivot_row[j]; // обновляем расстояние
                }
            }
        }

        // собираем обновлённые строки всех процессов в global_matrix
        MPI_Allgather(local_matrix.data(), local_rows * N, MPI_DOUBLE,
                      global_matrix.data(), local_rows * N, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        // обновляем локальные данные для следующей итерации
        local_matrix = std::vector<double>(global_matrix.begin() + displs[rank],
                                           global_matrix.begin() + displs[rank] + local_rows * N);
    }

    // собираем окончательную матрицу на rank 0
    MPI_Gatherv(local_matrix.data(), local_rows * N, MPI_DOUBLE,
                adjacency.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();                   // конец измерения времени

    if (rank == 0) {                          // выводим результаты на главном процессе
        std::cout << "Shortest paths matrix:" << std::endl;
        for (int i = 0; i < N; i++) {            // вывод строк
            for (int j = 0; j < N; j++) {           // вывод столбцов
                if (adjacency[i * N + j] >= INF) std::cout << "INF "; // INF вместо больших чисел
                else std::cout << adjacency[i * N + j] << " ";     // обычный вес
            }
            std::cout << std::endl;             // переход на новую строку
        }

        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl; // вывод времени
    }

    MPI_Finalize();                    // завершение работы MPI
    return 0;
}
