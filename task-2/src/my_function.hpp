#pragma once

#include <complex>

using namespace std;
using namespace std::complex_literals;

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