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
    vector<complex<double>> u_old(total_points), u_new(total_points);

    // Pivot indices for LAPACKE_dgbsv
    vector<lapack_int> ipiv(n);
    // Storage for tridiagonal matrix for solving banded equation
    // Initialize once and reuse for each time step and iteration
    // 4 rows because we have three diagonals and lapack needs
    // one extra for pivoting (first row)?
    vector<complex<double>> tridiagonal(4 * total_points);

    // Main diagonal
    for (int i = 0; i < total_points; ++i)
    {
        // Second row, no shift
        int main_diag_index = i + total_points * 2;
        // Ghost points, different coeficients
        if (i == 0 || i == total_points - 1)
            tridiagonal[main_diag_index] = 2i / (tau * h * h) - 1;
        else
            tridiagonal[main_diag_index] = 2.0 * (1i / (tau * h * h) - 1);
    } 

    // Fill upper and lower diagonal
    for (int i = 0; i < total_points - 1; ++i)
    {
        int upper_index = i + 1 + total_points * 1;
        tridiagonal[upper_index] = 1;

        int lower_index = i + total_points * 3;
        tridiagonal[lower_index] = 1;
    } 

    // Initial condition
    for (int i = 0; i < N + 1; ++i)
    {
        const double x = static_cast<double>(i) / N;
        // Time step is zero, fill from start
        solution[i] = cos(2 * numbers::pi * x) * exp(1i * numbers::pi / 4.0);
    }

    constexpr lapack_int kl = 1;  // number of subdiagonals
    constexpr lapack_int ku = 1;  // number of superdiagonals
    constexpr lapack_int nrhs = 1; // number of right-hand sides

    for (int step = 0; step < total_time_steps; ++step)
    {
        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // https://www.netlib.org/lapack/explore-html/db/df8/group__gbsv_gaff55317eb3aed2278a85919a488fec07.html#gaff55317eb3aed2278a85919a488fec07
            lapack_int info = LAPACKE_dgbsv(
                LAPACK_COL_MAJOR, 
                total_points, 
                1, 
                1, 
                1,
                tridiagonal.data(), 
                ldab, 
                ipiv.data(),
                b.data(), 
                total_points
            );
        }
    }
}