#include <iostream>
#include <complex>
#include <vector>
#include <span>
#include <omp.h>
#include <mpi.h>
#include <fstream>
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

double calculate_max_norm(solver_config_t const& config, istream &is);
vector<tuple<string, int>> get_assigned_configs(int argc, char **argv, int world_rank, int world_size);

int main( int argc, char **argv )
{
    if (argc < 2)
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
    string log_filename = "../logs/log_rank_" + to_string(world_rank) + ".log";
    auto file_logger = spdlog::basic_logger_mt("file_logger", log_filename);
    
    if (!file_logger) {
        cerr << "Failed to create file logger for rank " << world_rank;
    } else {
        file_logger->info("Logger initialized successfully for rank {0}", world_rank);
    }
    file_logger->set_level(spdlog::level::info);

    if (world_rank == 0)
    {
        file_logger->info("MPI initialized, world size: {0}", world_size);
    }

    openblas_set_num_threads(omp_get_max_threads());
    file_logger->info("[rank: {0}] OpenBLAS threads set to {1}", world_rank, omp_get_max_threads());

    // Config loading and solving
    vector<tuple<string, int>> filename_index_pairs = get_assigned_configs(argc, argv, world_rank, world_size);

    file_logger->info("[rank: {0}] Assigned {1} configurations in total", world_rank, filename_index_pairs.size());
    for(int i = 0; i < filename_index_pairs.size(); ++i)
    {
        const string config_filename = get<0>(filename_index_pairs[i]);
        const int config_id = get<1>(filename_index_pairs[i]);

        file_logger->info(
            "[rank: {0}] Assigned configuration {1} [id: {2}]", 
            world_rank, 
            config_filename, 
            config_id
        );

        solver_config_t config;
        ifstream is(config_filename);

        try {
            cereal::JSONInputArchive archive(is);
            archive(cereal::make_nvp("solver_config", config));
            is.close();
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
        const string sol_filename = "data-" + to_string(config_id) + ".bin";
        ofstream os(sol_filename, std::ios::binary);

        auto func = [&config](double x, double t){ return f(x, t, config.beta); };

        schrodinger_solver_t solver(config, config_id, func, os, file_logger, world_rank);

        // Solve
        solver.solve(u0);

        // Cleanup
        os.close();

        // log the maximum difference from the exact solution
        // throughout all space and time
        is = ifstream(sol_filename, std::ios::binary);
        
        const double max_norm = calculate_max_norm(config, is);

        is.close();

        file_logger->info("[id: {0}] max_norm = {1}", config_id, max_norm);
    }


    MPI_Finalize();

    return 0;
}

vector<tuple<string, int>> get_assigned_configs(int argc, char **argv, int world_rank, int world_size)
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

    vector<tuple<string, int>> filename_index_pairs(configs_for_this_rank);

    int rank_offset = world_rank <= remaining
        ? world_rank * (configs_per_rank + 1)
        : remaining * (configs_per_rank + 1) + (world_rank - remaining) * configs_per_rank;

    for (int i = 0; i < configs_for_this_rank; ++i)
    {
        filename_index_pairs[i] = { string(argv[rank_offset + 1 + i]), rank_offset + i };
    }
    
    return filename_index_pairs;
}

double calculate_max_norm(solver_config_t const& config, istream &is)
{
    vector<complex<double>> u(config.total_points);
    double max_norm = 0;
    for (int step = 0; step < config.total_time_steps; ++step)
    {
        is.read(
            reinterpret_cast<char*>(u.data()),
            config.total_points * sizeof(complex<double>)
        );

        const double t = step * config.tau;
        for (int i = 0; i < config.total_points; ++i)
        {
            const double x = i * config.h;
            max_norm = max(max_norm, abs(u[i] - u_exact(x, t)));
        }
    }
    return max_norm;
}
