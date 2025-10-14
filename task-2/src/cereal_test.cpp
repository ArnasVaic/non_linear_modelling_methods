#include <fstream>

#include <cereal/types/complex.hpp>
#include <cereal/archives/json.hpp>

#include "solver_config.hpp"

int main()
{
  solver_config_t cfg;
  cfg.max_iterations = 100;
  cfg.delta = 1e-5;
  cfg.N = 299;
  cfg.beta = 1.0;
  cfg.T = 10.0;
  cfg.tau = 0.01;

  {
    std::ofstream os("test.json");
    cereal::JSONOutputArchive {os}(cereal::make_nvp("solver_config", cfg));
  }
}