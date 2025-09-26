#include <iostream>
#include <complex>
#include <vector>
#include <numbers>
#include <mpi.h>

extern "C" {
    #include <lapacke.h>
}

using namespace std;

complex<double> f(double x, double t)
{
    return 0;
}

int main( int argc, char **argv )
{
    // Constants
    constexpr double max_iterations = 100;
    constexpr double beta = 0.1;
    constexpr double delta = 10e-5;
    
    constexpr int N = 999;
    constexpr int total_points = N + 1;
    constexpr double h = 1 / static_cast<double>(N);

    constexpr double T = 1;
    constexpr double tau = 0.0001;
    constexpr int total_time_steps = static_cast<int>(T / tau);

    vector<complex<double>> solution(total_time_steps * total_points);

    // Initial condition
    for (int i = 0; i < N + 1; ++i)
    {
        const double x = static_cast<double>(i) / N;
        // Time step is zero, fill from start
        solution[i] = cos(2 * numbers::pi * x) * exp(1il * numbers::pi / 4);
        std::cout << solution[i] << '\n';
    }
}