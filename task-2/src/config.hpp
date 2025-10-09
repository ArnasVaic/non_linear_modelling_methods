#pragma once

constexpr int max_iterations = 100;
constexpr double _beta = 10.0;
constexpr double delta = 1e-5;

constexpr int N = 999;
constexpr int total_points = N + 1;
constexpr double h = 1 / static_cast<double>(N);
constexpr double h2 = h * h;

constexpr double T = 10;
constexpr double tau = 0.001;
constexpr int total_time_steps = static_cast<int>(T / tau);