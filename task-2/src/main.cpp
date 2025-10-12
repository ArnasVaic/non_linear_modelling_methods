#include <iostream>
#include <complex>
#include <vector>
#include <span>
#include <omp.h>
#include <mpi.h>
#include <fstream>
#include <format>
#include <algorithm>
#include <cblas.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <cereal/archives/json.hpp>

#include "solver_config.hpp"
#include "my_function.hpp"
#include "schrodinger_solver.hpp"

using namespace std;
using namespace std::complex_literals;

vector<string> get_assigned_configs(int argc, char **argv, int world_rank, int world_size)
{
    // First arg is reserved for initial condition file
    int config_cnt = argc - 1;
    int configs_per_rank = config_cnt / world_size;

    // remainder of configs that can be split evenly amongst the ranks
    int remaining = config_cnt % world_size;

    // First 'remaining' ranks get one extra config
    int configs_for_this_rank = world_rank < remaining
        ? configs_per_rank + 1
        : configs_per_rank;

    vector<string> filenames(configs_for_this_rank);

    int rank_offset = world_rank <= remaining
        ? world_rank * (configs_per_rank + 1)
        : remaining * (configs_per_rank + 1) + (world_rank - remaining) * configs_per_rank;

    for (int i = 0; i < configs_for_this_rank; ++i)
    {
        filenames[i] = string(argv[rank_offset + 1 + i]);
    }
    
    return filenames;
}

int main( int argc, char **argv )
{
    // 1 arg is for initial condition, others are for solver configurations
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " config_file { config_file }" << endl;
        return 1;
    }
    
    // MPI intialization
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Logger configuration
    string log_filename = format("logs/log_rank_{}.log", world_rank);
    auto file_logger = spdlog::basic_logger_mt("file_logger", log_filename);
    
    if (!file_logger) {
        cerr << "Failed to create file logger for rank " << world_rank;
    } else {
        file_logger->info("Logger initialized successfully for rank {0}", world_rank);
    }
    file_logger->set_level(spdlog::level::info);

    if (world_rank == 0)
    {
        file_logger->info("MPI initialized with world size {0}", world_size);
    }

    openblas_set_num_threads(omp_get_max_threads());
    file_logger->info("rank = {0}, OpenBLAS threads set to {1}", world_rank, omp_get_max_threads());

    // Config loading and solving
    vector<string> filenames = get_assigned_configs(argc, argv, world_rank, world_size);

    file_logger->info("Rank {0} assigned {1} configurations in total", world_rank, filenames.size());
    for(int i = 0; i < filenames.size(); ++i)
    {
        file_logger->info("Rank {0} assigned configuration {1}", world_rank, filenames[i]);

        solver_config_t config;
        ifstream is(filenames[i]);

        try {
            cereal::JSONInputArchive archive(is);
            archive(config);
        } catch (cereal::RapidJSONException& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            std::exit(1);
        }
        
        // initial condition
        vector<complex<double>> u0(config.total_points);
        for(int i = 0; i < config.total_points; ++i)
        {
            const double x = i * config.h;
            u0[i] = u_exact(x, 0);
        }

        // Setup solver
        string sol_filename = format("rank_{}_config_{}.bin", world_rank, i);
        ofstream os(sol_filename, std::ios::binary);

        auto func = [&config](double x, double t){ return f(x, t, config.beta); };

        schrodinger_solver_t solver(config, func, os, file_logger, world_rank);

        // Solve
        solver.solve(u0);

        // Cleanup
        os.close();
    }


    MPI_Finalize();

    return 0;
}
