#include <iostream>
#include <complex>
#include <vector>
#include <numbers>
#include <chrono>
#include <span>
#include <omp.h>
#include <fstream>
#include <algorithm>
#include <cblas.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

extern "C" {
    #include <lapacke.h>
}

#include "config.hpp"
#include "my_function.hpp"

using namespace std;
using namespace std::complex_literals;

// f(x, t) = ∂/∂t u - i ∂²/∂x² u - β ∂/∂x (|u²|u)
inline complex<double> f(double x, double t)
{
    return Dt_u(x, t) - 1.0i * Dxx_u(x, t) - _beta * Dx_u2u(x, t);
}

void initialize_rhs(
    span<complex<double>> u_old,
    span<complex<double>> u,
    span<complex<double>> rhs,
    int step
);

int main( int argc, char **argv )
{
    // Create a logger without throwing
    auto file_logger = spdlog::basic_logger_mt("file_logger", "build/log.txt");
    
    if (!file_logger) {
        std::cerr << "Failed to create file logger!" << std::endl;
    } else {
        file_logger->info("================================================================");
        file_logger->info("Logger initialized successfully");
    }

    file_logger->set_level(spdlog::level::info);

    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " initial_condition_file1 [initial_condition_file2 ...]" << endl;
        return 1;
    }

    const int nrhs = argc - 1; // number of initial condition files
    file_logger->info("number of initial conditions: {0}", nrhs);

    openblas_set_num_threads(omp_get_max_threads());
    file_logger->info("OpenBLAS threads set to {}", omp_get_max_threads());
    file_logger->info("max_iterations: {0}", max_iterations);
    file_logger->info("beta: {0}", _beta);
    file_logger->info("delta: {0}", delta);
    file_logger->info("N: {0}", N);
    file_logger->info("h: {0}", h);
    file_logger->info("tau: {0}", tau);
    file_logger->info("total_time_steps: {0}", total_time_steps);

    // manual stop flag
    bool stop_simulation = false;

    file_logger->info("Initializing tridiagonal matrix diagonals");

    // last time step solution
    vector<complex<double>> u(nrhs * total_points);       
    // Incremental approximation storage
    vector<complex<double>> u_old(nrhs * total_points);

    // Tridiagonal matrix
    vector<complex<double>> super_diagonal(total_points - 1);
    fill(super_diagonal.begin(), super_diagonal.end(), 1.0);
    vector<complex<double>> super_diagonal_backup(super_diagonal);

    vector<complex<double>> sub_diagonal(total_points - 1);
    fill(sub_diagonal.begin(), sub_diagonal.end(), 1.0);
    vector<complex<double>> sub_diagonal_backup(sub_diagonal);

    vector<complex<double>> diagonal(total_points);
    fill(diagonal.begin(), diagonal.end(), 2i * h2 / tau - 2.0);
    diagonal[0]                 = 2i * h2 / tau - 1.0;
    diagonal[total_points - 1]  = 2i * h2 / tau - 1.0;
    vector<complex<double>> diagonal_backup(diagonal);

    // Right hand side vector
    vector<complex<double>> rhs(nrhs * total_points);

    file_logger->info("Initializing initial condition vector");

    vector<ofstream> output_file_handles(nrhs);

    // Read initial conditions
    for (int k = 0; k < nrhs; ++k)
    {
        const string filename = argv[k + 1];
        ifstream is(filename, ios::binary);
        if (!is)
        {
            file_logger->info("Failed to open initial condition file: {0}", filename);
            return 1;
        }

        // Read exactly total_points elements into column k
        is.read(reinterpret_cast<char*>(&u[k * total_points]), total_points * sizeof(complex<double>));
        if (!is)
        {
            file_logger->info("Error reading data from file: {0}", filename);
            return 1;
        }

        is.close();

        // additionally, create a result file
        output_file_handles[k] = ofstream("solution-" + filename, ios::binary);

        // Write first line
        output_file_handles[k].write(
            reinterpret_cast<const char*>(&u[k * total_points]), 
            total_points * sizeof(complex<double>)
        );
    }

    // helper for slices
    auto slice = [&](auto& vec, int k) {
        return std::span<std::complex<double>>(vec.data() + k * total_points, total_points);
    };

    vector<bool> rhs_converged(nrhs);

    auto start = std::chrono::high_resolution_clock::now();

    // when step is t, we are calculating solution at t + 1
    for (int step = 0; step < total_time_steps - 1; ++step)
    {
        const int time_step_offset = step * total_points;

        u_old.assign(u.begin(), u.end());

        fill(rhs_converged.begin(), rhs_converged.end(), false);

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // Recalculate RHS vectors
            for (int k = 0; k < nrhs; ++k)
            {
                const int rhs_offset = k * total_points; 
                initialize_rhs(slice(u_old, k), slice(u, k), slice(rhs, k), step);
            }

            // Solve systems of equations
            lapack_int info = LAPACKE_zgtsv(
                LAPACK_COL_MAJOR,
                total_points, 
                nrhs, 
                reinterpret_cast<lapack_complex_double*>(sub_diagonal.data()),
                reinterpret_cast<lapack_complex_double*>(diagonal.data()),
                reinterpret_cast<lapack_complex_double*>(super_diagonal.data()),
                reinterpret_cast<lapack_complex_double*>(rhs.data()),
                total_points
            );

            if (info != 0) {
                // Early exit
                file_logger->error("LAPACKE_zgtsv failed, info = {0}", info);
                stop_simulation = true;
                break;
            }

            // Reinitialize tridiagonal containers (ChatGPT says they might be overwriten...)
            sub_diagonal.assign(sub_diagonal_backup.begin(), sub_diagonal_backup.end());
            super_diagonal.assign(super_diagonal_backup.begin(), super_diagonal_backup.end());
            diagonal.assign(diagonal_backup.begin(), diagonal_backup.end());

            for (int k = 0; k < nrhs; ++k)
            {
                // enforce boundary conditions
                const int rhs_offset = k * total_points;
                rhs[rhs_offset + 0] = rhs[rhs_offset + 1];
                rhs[rhs_offset + total_points - 1] = rhs[rhs_offset + total_points - 2];

                // Each initial condition might need different number of iterations

                if (rhs_converged[k])
                {
                    continue;
                }

                double max_norm = 0.0;
                for (int i = 0; i < total_points; ++i)
                {
                    max_norm = max(max_norm, abs(rhs[rhs_offset + i] - u_old[rhs_offset + i]));
                }

                if (max_norm < delta)
                {
                    file_logger->debug("rhs = {3}, step = {0}, iter = {1}, max_norm = {2} converged", step, iter, max_norm, k);
                    rhs_converged[k] = true;
                }
                else
                {
                    if (iter == max_iterations - 1)
                    {
                        file_logger->warn("rhs = {3}, step = {0}, iter = {1}, max_norm = {2} reached max iterations", step, iter, max_norm, k);
                    }
                    else
                    {
                        file_logger->debug("rhs = {3}, step = {0}, iter = {1}, max_norm = {2}", step, iter, max_norm, k);   
                    }
                }
            }

            if ( iter == max_iterations - 1 || all_of(rhs_converged.begin(), rhs_converged.end(), [](bool v) { return v; }) )
            {
                break; // no need to keep going through iterations if all solutions converged
            }

            swap(u_old, rhs);
        }

        u.assign(rhs.begin(), rhs.end());

        for (int k = 0; k < nrhs; ++k)
        {
            const int rhs_offset = k * total_points;
            output_file_handles[k].write(
                reinterpret_cast<const char*>(&rhs[k * total_points]), 
                total_points * sizeof(complex<double>)
            );
        }
        
        if (step % 100 == 0) file_logger->flush();

        if (stop_simulation)
        {
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    file_logger->info("Solve time (seconds): {0}", elapsed.count());
    file_logger->info("Solve time per initial condition (seconds): {0}", elapsed.count() / nrhs);

    for (int k = 0; k < nrhs; ++k)
    {
        file_logger->info("Results written to {0}",  "solution-" + string(argv[k + 1]));
        output_file_handles[k].close();
    }
}

void initialize_rhs(
    span<complex<double>> u_old,
    span<complex<double>> u,
    span<complex<double>> rhs,
    int step
)
{
    #pragma omp parallel for
    for (int i = 0; i < total_points; ++i)
    {
        // Index for laplacian with neumann boundary conditions
        // we have a simplistic rule that u_0 = u_1 & u_N = u_{N-1}
        const int il = i ==                0 ?                1 : i - 1;
        const int ir = i == total_points - 1 ? total_points - 2 : i + 1;
        
        // Precompute some results from non-linear first order part
        const complex<double> half_u_r = 0.5 * (u_old[ir] + u[ir]);
        const complex<double> half_u_l = 0.5 * (u_old[il] + u[il]);

        const complex<double> half_u_r_abs = abs(half_u_r);
        const complex<double> half_u_l_abs = abs(half_u_l);

        const double x = static_cast<double>(i) / N;

        rhs[i] =

            // Single u_j part from the time derivative approx.
            2i * h2 / tau * u[i] +

            // Laplacian part
            - u[il] + 2.0 * u[i] - u[ir] +

            // Non-linear first order part - β ∂/∂x |u|²u
            1i * h * _beta * (
                (half_u_r_abs * half_u_r_abs * half_u_r) -
                (half_u_l_abs * half_u_l_abs * half_u_l)
            ) +

            // Function part
            1i * h2 * ( f(x, tau * step) + f(x, tau * (step + 1)) );
    }
}