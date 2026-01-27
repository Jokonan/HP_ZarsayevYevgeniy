#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath> // для sqrt

int main() {
    const int N = 10000000; // размер массива
    std::vector<double> data(N, 1.0); // заполняем массив единицами

    // список потоков для теста
    std::vector<int> threads_list = {1, 2, 4, 8};

    for (int num_threads : threads_list) { // пробегаемся по разному числу потоков
        double sum = 0.0; // для суммы
        double mean = 0.0; // для среднего
        double variance = 0.0; // для дисперсии

        omp_set_num_threads(num_threads); // задаем число потоков

        double t1 = omp_get_wtime(); // старт таймера
        #pragma omp parallel for reduction(+:sum) // параллельный цикл с редукцией суммы
        for (int i = 0; i < N; i++) {
            sum += data[i]; // суммируем элементы
        }
        double t2 = omp_get_wtime(); // конец таймера

        mean = sum / N; // считаем среднее

        double t3 = omp_get_wtime(); // старт таймера для дисперсии
        #pragma omp parallel for reduction(+:variance) // параллельный цикл с редукцией дисперсии
        for (int i = 0; i < N; i++) {
            variance += (data[i] - mean) * (data[i] - mean); // считаем квадраты отклонений
        }
        double t4 = omp_get_wtime(); // конец таймера

        variance /= N; // окончательная дисперсия

        // выводим результаты для текущего числа потоков
        std::cout << "Threads: " << num_threads 
                  << " | Sum: " << sum 
                  << ", Mean: " << mean 
                  << ", Variance: " << variance 
                  << " | Time sum: " << (t2 - t1) 
                  << " sec | Time variance: " << (t4 - t3) 
                  << " sec | Total parallel time: " << ((t2-t1)+(t4-t3)) << " sec\n";
    }

    return 0;
}
