#pragma once

#include <string>
#include <complex>
#include <vector>
#include <numbers>
#include <chrono>
#include <span>

extern "C" {
    #include <lapacke.h>
}

#include "spdlog/spdlog.h"
#include "solver_config.hpp"

using namespace std::complex_literals;

template<typename Func>
struct schrodinger_solver_t
{
    // Solver configuration
    solver_config_t config;

    // Solver configuration index
    int config_index;

    // Specific solver function 
    Func f;

    // World rank of this solver
    int world_rank;

    // Tridiagonal matrix storage
    vector<complex<double>> sup_diag;
    vector<complex<double>> diagonal;
    vector<complex<double>> sub_diag;

    // LAPACK overrides passed tridiagonals 
    // so keeping copies is required
    vector<complex<double>> sup_diag_copy;
    vector<complex<double>> diagonal_copy;
    vector<complex<double>> sub_diag_copy;

    // Numerical solution values at times t and t + tau
    vector<complex<double>> u;
    vector<complex<double>> u_next;

    // Naming might be confusing...
    // The non-linear part that after few iterations 
    // should converge to the u_next
    vector<complex<double>> u_old;

    // Output sink for solution
    ostream &os;

    // Loggers
    shared_ptr<spdlog::logger> logger;

    schrodinger_solver_t(
        solver_config_t config,
        int config_index,
        Func f,
        ostream &os,
        shared_ptr<spdlog::logger> logger,
        int world_rank)
    : f(f)
    , os(os)
    , config(config)
    , config_index(config_index)
    , logger(logger)
    , world_rank(world_rank)
    {
        initialize_tridiagonals();

        u.resize(config.total_points);
        u_next.resize(config.total_points);
        u_old.resize(config.total_points);

        // log config values
        logger->info("[id: {0}] max_iterations: {1}", config_index , config.max_iterations);
        logger->info("[id: {0}] beta: {1}", config_index, config.beta);
        logger->info("[id: {0}] delta: {1}", config_index, config.delta);
        logger->info("[id: {0}] N: {1}", config_index, config.N);
        logger->info("[id: {0}] h: {1}", config_index, config.h);
        logger->info("[id: {0}] tau: {1}", config_index, config.tau);
        logger->info("[id: {0}] total_time_steps: {1}", config_index, config.total_time_steps);
    }

    void initialize_tridiagonals()
    {
        // Non main diagonals have one less elements
        sup_diag.resize(config.total_points - 1);
        sub_diag.resize(config.total_points - 1);
        diagonal.resize(config.total_points);

        // Non main diagonals are ones
        fill(sup_diag.begin(), sup_diag.end(), 1.0);
        fill(sub_diag.begin(), sub_diag.end(), 1.0);

        // Main diagonal is more complicated and requires
        // Different expressions for first and last elements
        const complex<double> main_diag_value 
            = 2i * config.h2 / config.tau - 2.0;

        const complex<double> main_diag_boundary_value 
            = 2i * config.h2 / config.tau - 1.0;

        fill(diagonal.begin(), diagonal.end(), main_diag_value);
        diagonal[0]                        = main_diag_boundary_value;
        diagonal[config.total_points - 1]  = main_diag_boundary_value;

        sup_diag_copy = sup_diag;
        sub_diag_copy = sub_diag;
        diagonal_copy = diagonal;
    }

    void solve(vector<complex<double>> const& u0)
    {
        u.assign(u0.begin(), u0.end());

        write_solution_step(u0);

        auto start = std::chrono::high_resolution_clock::now();

        // when step is t, we are calculating solution at t + tau
        for (int step = 0; step < config.total_time_steps - 1; ++step)
        {
            // Make sure we begin approximating non-linear 
            // part from the current value of u
            u_old.assign(u.begin(), u.end());

            lapack_int info = solve_step(step);
            if (info != 0)
            {
                break;
            }

            // After solving for u_next, assign it to u and repeat
            u.assign(u_next.begin(), u_next.end());

            write_solution_step(u_next);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        logger->info("[id: {0}] Solve time (seconds): {1}", config_index, elapsed.count());
    }

    inline lapack_int solve_system()
    {
        const lapack_int info = LAPACKE_zgtsv(
            LAPACK_COL_MAJOR,
            config.total_points, 
            1, 
            reinterpret_cast<lapack_complex_double*>(sub_diag.data()),
            reinterpret_cast<lapack_complex_double*>(diagonal.data()),
            reinterpret_cast<lapack_complex_double*>(sup_diag.data()),
            // u_next needs to be filled with values for the RHS of the quation
            reinterpret_cast<lapack_complex_double*>(u_next.data()),
            config.total_points
        );

        // Reassign original tridiagonal values
        sub_diag.assign(sub_diag_copy.begin(), sub_diag_copy.end());
        sup_diag.assign(sup_diag_copy.begin(), sup_diag_copy.end());
        diagonal.assign(diagonal_copy.begin(), diagonal_copy.end());

        return info;
    }

    inline lapack_int solve_step(int step)
    {
        // Iterate until u_old converges to u_next
        for (int iteration = 0; iteration < config.max_iterations; ++iteration)
        {
            initialize_rhs(step);

            lapack_int info = solve_system();
            
            if (info != 0) {
                logger->error(
                    "[id: {0}, step: {1}, it: {2}] LAPACKE_zgtsv failed, info = {3}", 
                    config_index,
                    step,
                    iteration,
                    info
                );
                return info;
            }

            enforce_boundary();

            auto [ converged, norm ] = check_convergance(step, iteration);
            log_convergance_info(converged, norm, step, iteration);

            if (converged)
                break;

            // After solving u_next will contain the solution for next step to
            // continue this iteration process we swap it to be the new u_old and repeat
            swap(u_old, u_next);
        }

        
        return 0;
    }

    inline void log_convergance_info(bool converged, double norm, int step, int iteration)
    {
        if (converged)
        {
            logger->debug(
                "[id: {0}, step: {1}, it: {2}] norm = {3}, converged", 
                config_index, step, iteration, norm
            );
            return;
        }

        if (iteration == config.max_iterations - 1)
        {
            logger->warn(
                "[id: {0}, step: {1}, it: {2}] norm = {3}, max iterations reached", 
                config_index, step, iteration, norm
            );
            return;
        }

        logger->debug(
            "[id: {0}, step: {1}, it: {2}] norm = {3}", 
            config_index, step, iteration, norm
        );
    }

    inline void initialize_rhs(int step)
    {
        #pragma omp parallel for
        for (int i = 0; i < config.total_points; ++i)
        {
            // Index for laplacian with neumann boundary conditions
            // we have a simplistic rule that u_0 = u_1 & u_N = u_{N-1}
            const int il = i == 0 
                ? 1 
                : i - 1;
            const int ir = i == config.total_points - 1 
                ? config.total_points - 2 
                : i + 1;
            
            // Precompute some results from non-linear first order part
            const complex<double> half_u_r = 0.5 * (u_old[ir] + u[ir]);
            const complex<double> half_u_l = 0.5 * (u_old[il] + u[il]);

            const complex<double> half_u_r_abs = abs(half_u_r);
            const complex<double> half_u_l_abs = abs(half_u_l);

            const double x = static_cast<double>(i) / config.N;

            u_next[i] =

                // Single u_j part from the time derivative approx.
                2i * config.h2 / config.tau * u[i] +

                // Laplacian part
                - u[il] + 2.0 * u[i] - u[ir] +

                // Non-linear first order part - β ∂/∂x |u|²u
                1i * config.h * config.beta * (
                    (half_u_r_abs * half_u_r_abs * half_u_r) -
                    (half_u_l_abs * half_u_l_abs * half_u_l)
                ) +

                // Function part
                1i * config.h2 * ( f(x, config.tau * step) + f(x, config.tau * (step + 1)) );
        }
    }

    inline void enforce_boundary()
    {
        u_next[0] = u_next[1];
        u_next[config.total_points - 1] = u_next[config.total_points - 2];
    }

    inline tuple<bool, double> check_convergance(int step, int iteration)
    {
        double max_norm = 0.0;
        for (int i = 0; i < config.total_points; ++i)
        {
            const double error = abs(u_next[i] - u_old[i]);
            max_norm = max(max_norm, error);
        }
        return { max_norm < config.delta, max_norm };
    }

    inline void write_solution_step(vector<complex<double>> const& solution_step)
    {
        os.write(
            reinterpret_cast<const char*>(solution_step.data()), 
            config.total_points * sizeof(complex<double>)
        );
    }
};
