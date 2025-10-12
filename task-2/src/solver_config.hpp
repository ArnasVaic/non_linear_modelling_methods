#pragma once

#include <cereal/types/complex.hpp>

struct solver_config_t
{
    // Maximum number of iterations for approximating non-linear solution parts
    size_t max_iterations;

    // Convergance threshold for non-linear solution parts
    double delta;

    // Number of discrete points - 1
    size_t N;

    // Coefficient for first order non-linear term  (∂/∂x |u|²u)
    double beta;

    // Distance between physical points
    double h;

    // Precomputed h * h
    double h2;

    // Total time of the simulation
    double T;

    // Time step
    double tau;

    // Number of discrete points
    size_t total_points;

    // Total number of time steps
    size_t total_time_steps;

    template<class Archive>
    void load(Archive& ar) {
        ar(
            max_iterations, 
            delta,
            N,
            beta,
            T,
            tau
        );

        total_points = N + 1;
        h = 1.0 / N;
        h2 = h * h;
        total_time_steps = static_cast<int>(T / tau);
    }
};
