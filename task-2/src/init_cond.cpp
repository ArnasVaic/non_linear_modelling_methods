#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>

#include "my_function.hpp"
#include "config.hpp"

using namespace std;
using namespace std::complex_literals;

int main()
{
    vector<complex<double>> u(total_points);

    // Fill initial condition
    for (int i = 0; i < total_points; ++i)
    {
        double x = static_cast<double>(i) / N;
        // u[i] = u_exact(x, 0);
        u[i] = 0.5 * (x - 0.5);
    }

    // Write to binary file
    const string filename = "build/initial_condition_linear.bin";
    ofstream os(filename, ios::binary);
    if (!os)
    {
        cerr << "Error opening file " << filename << " for writing!" << endl;
        return 1;
    }

    os.write(reinterpret_cast<const char*>(u.data()), u.size() * sizeof(complex<double>));
    os.close();

    cout << "Initial condition written to " << filename << endl;

    return 0;
}
