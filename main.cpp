#include "betabernoulli_mcts.h"
#include <iostream>

int main() {
    std::cout << "Starting BetaMCTS LeelaIntegration example..." << std::endl;
    
    BetaMCTS::LeelaIntegration leela_integration_instance;
    leela_integration_instance.run_simulation_example();
    
    std::cout << "BetaMCTS LeelaIntegration example finished." << std::endl;
    return 0;
}
