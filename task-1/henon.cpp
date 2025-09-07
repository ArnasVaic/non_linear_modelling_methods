#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <math.h>
#include <chrono>
#include <mpi.h>

using namespace std;

// designed to run on HPC
int main(int argc, char **argv) {
    float left = -5.0f;
    float right = 5.0f;
    float top = 5.0f;
    float bottom = -5.0f;

    constexpr float alpha = 0.2f;
    constexpr float beta = 1.01f;
    constexpr float target = 100.0f;
    constexpr int max_iter_count = 1000;

    constexpr int nx = 1000;
    constexpr int ny = 1000;
    constexpr int total = nx * ny;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_rank = ny / size;
    int remainder = ny % size;

    int start_row, end_row;

    if (rank < remainder) {
        // Split vertical lines as evenly as possible
        // but in case theres a remainder, give one extra
        // line to each thread until no more are left. That 
        // way the busiest thread will have at most 1 extra 
        // vertical line to process compared to the thread 
        // with the least work assigned
        start_row = rank * (rows_per_rank + 1);
        end_row = start_row + rows_per_rank + 1;

    } else {
        // the number of ranks that got an extra row will be equal to the remainder
        // so account for the offset of previous ranks. Then the rest of the ranks
        // after the remainder will get the normal number of rows.
        start_row = remainder * (rows_per_rank + 1) + (rank - remainder) * rows_per_rank;
        end_row = start_row + rows_per_rank;
    }

    int local_count = (end_row - start_row) * nx;

    cout << "thread[" << rank << "] is assigned " << local_count << " rows.\n";

    std::vector<int> local_result(local_count);

    auto start = std::chrono::high_resolution_clock::now();

    for (int j = start_row; j < end_row; ++j) {
        for (int i = 0; i < nx; ++i) {
            float x0 = left + (right - left) * i / (nx - 1);
            float y0 = bottom + (top - bottom) * j / (ny - 1);

            float x_new = x0;
            float y_new = y0;

            float x_old = x0;
            float y_old = y0;

            int iteration = 0;
            while (true) {

                x_new = 1.0f - alpha * x_old * x_old + y_old;
                y_new = beta * x_old;

                if (fabs(x_new) + fabs(y_new) > target || iteration == max_iter_count)
                {
                    int local_index = (j - start_row) * nx + i; // index in local array
                    local_result[local_index] = iteration;
                    break;   
                }

                ++iteration;

                x_old = x_new;
                y_old = y_new;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::vector<int> sendcounts, displs;
    if (rank == 0) {
        sendcounts.resize(size);
        displs.resize(size);

        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_start, r_end;

            // The same formulas that were used to calculate r_start and r_end
            // in individual threads. We need to do it again because this is 
            // the collection thread which everybody is sending their results to.
            if (r < remainder) {
                r_start = r * (rows_per_rank + 1);
                r_end = r_start + rows_per_rank + 1;
            } else {
                r_start = remainder * (rows_per_rank + 1) + (r - remainder) * rows_per_rank;
                r_end = r_start + rows_per_rank;
            }
            sendcounts[r] = (r_end - r_start) * nx;
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    std::vector<int> gathered_result;
    if (rank == 0) gathered_result.resize(total);

    MPI_Gatherv(
        local_result.data(),
        local_count,
        MPI_INT,
        gathered_result.data(),
        sendcounts.data(),
        displs.data(),
        MPI_INT, 
        0, 
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        std::cout << "Time taken: " << elapsed.count() << " seconds\n";
        ofstream output;
        output.open("result.txt");

        for (int i = 0; i < total; ++i) {
            output << gathered_result[i] << '\n';
        }

        output.close();
    }
    MPI_Finalize();
}