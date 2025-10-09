#include <iostream>
#include <complex>
#include <vector>
#include <numbers>
#include <chrono>
#include <span>
#include <omp.h>
#include <fstream>
#include <cblas.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

extern "C" {
    #include <lapacke.h>
}

using namespace std;
using namespace std::complex_literals;

// Constants
constexpr int max_iterations = 100;
constexpr double _beta = 10.0;
constexpr double delta = 1e-5;

constexpr int N = 999;
constexpr int total_points = N + 1;
constexpr double h = 1 / static_cast<double>(N);
constexpr double h2 = h * h;

constexpr double T = 1;
constexpr double tau = 0.0001;
constexpr int total_time_steps = static_cast<int>(T / tau);

// constructing the f(x, t) so that it would fit the exact solution:
// u(x, t) parameters
constexpr double A = 0.1;
constexpr double w0 = 100.0;
constexpr double sigma = 0.09;
constexpr double x0 = 0.5;

// B(x)
inline double B(double x)
{
    const double xminx0 = x - x0;
    return exp(- xminx0 * xminx0 / (4.0 * sigma * sigma));
}

// ∂/∂x B(x)
inline double Dx_B(double x)
{
    const double c = - 1.0 / (2.0 * sigma * sigma);
    return c * B(x) * (x - x0);
}

// ∂²/∂x² B(x)
inline double Dxx_B(double x)
{
    const double c = - 1.0 / (2.0 * sigma * sigma);
    return c * (Dx_B(x) * (x - x0) + B(x));
}

// W(x)
inline complex<double> W(double x) { return exp(1.0i * w0 * x); }
// ∂/∂x W(x)
inline complex<double> Dx_W(double x) { return 1.0i * w0 * exp(1.0i * w0 * x); }
// ∂²/∂x² W(x)
inline complex<double> Dxx_W(double x) { return - w0 * w0 * exp(1.0i * w0 * x); }

// M(t)
inline complex<double> M(double t) { return exp(- 1.0i * t); }
// ∂/∂t M(t)

// my exact solution
inline complex<double> u_exact(double x, double t)
{
    return A * B(x) * W(x) * M(t);
}

// ∂/∂t u
inline complex<double> Dt_u(double x, double t)
{
    return -1.0i * u_exact(x, t);
}

// ∂²/∂x² u
inline complex<double> Dxx_u(double x, double t)
{
    return A * M(t) * (Dxx_B(x) * W(x) + 2.0 * Dx_B(x) * Dx_W(x) + B(x) * Dxx_W(x));
}

// ∂/∂x |u²|u <- the first order part
inline complex<double> Dx_u2u(double x, double t)
{
    return A * A * A * M(t) * ( 3 * B(x) * B(x) * Dx_B(x) * W(x) + B(x) * B(x) * B(x) * Dx_W(x));
}

// f(x, t) = ∂/∂t u - i ∂²/∂x² u - β ∂/∂x (|u²|u)
inline complex<double> f(double x, double t)
{
    return Dt_u(x, t) - 1.0i * Dxx_u(x, t) - _beta * Dx_u2u(x, t);
}

int main( int argc, char **argv )
{
    // Create a logger without throwing
    auto file_logger = spdlog::basic_logger_mt("file_logger", "build/log.txt");
    
    if (!file_logger) {
        std::cerr << "Failed to create file logger!" << std::endl;
    } else {
        file_logger->info("Logger initialized successfully");
    }

    file_logger->set_level(spdlog::level::info);

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

    // last time step solution
    vector<complex<double>> u(total_points);       
    // Incremental approximation storage
    vector<complex<double>> u_old(total_points), u_new(total_points);

    file_logger->info("Initializing tridiagonal matrix diagonals");
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

    vector<complex<double>> rhs(total_points);

    file_logger->info("Initializing initial condition vector");

    // Initial condition
    for (int i = 0; i < total_points; ++i)
    {
        const double x = static_cast<double>(i) / N;
        u[i] = u_exact(x, 0);
    }

    const string output_filename = "build/data.bin";
    ofstream os(output_filename, ios::binary);

    os.write(
        reinterpret_cast<const char*>(u.data()), 
        u.size() * sizeof(complex<double>)
    );

    auto start = std::chrono::high_resolution_clock::now();

    // when step is t, we are calculating solution at t + 1
    for (int step = 0; step < total_time_steps - 1; ++step)
    {
        const int time_step_offset = step * total_points;

        u_old.assign(u.begin(), u.end());

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            // Recalculate RHS vector
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

            lapack_int info = LAPACKE_zgtsv(
                LAPACK_COL_MAJOR,
                total_points, 
                1, 
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

            u_new.assign(rhs.begin(), rhs.end());

            // enforce boundary conditions, no way this is the reason it diverges...
            u_new[0] = u_new[1];
            u_new[total_points - 1] = u_new[total_points - 2];

            double max_norm = 0.0;
            for (int i = 0; i < total_points; ++i)
            {
                max_norm = max(max_norm, abs(u_new[i] - u_old[i]));
            }

            if (max_norm < delta)
            {
                file_logger->debug("step = {0}, iter = {1}, max_norm = {2} converged", step, iter, max_norm);
                break;
            }
            else if (iter == max_iterations - 1)
            {
                file_logger->warn("step = {0}, iter = {1}, max_norm = {2} reached max iterations", step, iter, max_norm);
            }
            else
            {
                file_logger->debug("step = {0}, iter = {1}, max_norm = {2}", step, iter, max_norm);   
            }

            u_old.assign(u_new.begin(), u_new.end());
            
        }

        u.assign(u_new.begin(), u_new.end());

        os.write(
            reinterpret_cast<const char*>(u_new.data()), 
            u_new.size() * sizeof(complex<double>)
        );

        if (step % 100 == 0) file_logger->flush();

        if (stop_simulation)
        {
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    file_logger->info("Solve time (seconds): {0}", elapsed.count());
    file_logger->info("Results written to {0}", output_filename);
    os.close();
}