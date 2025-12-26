#include "rerun/archetypes/scalars.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>

#include <rerun.hpp>
#include "robot_interface.hpp"


int main(int argc, char const *argv[]) {
    std::cout << "Hello, World!" << std::endl;
    
    std::cout << "Initializing G1Interface..." << std::endl;
    std::string networkInterface = "enp58s0";
    auto g1Interface = G1HarwareInterface(networkInterface);
    std::cout << "G1HarwareInterface initialized successfully" << std::endl;
    
    const rerun::RecordingStream rec = rerun::RecordingStream("rerun_example_cpp");
    // Try to spawn a new viewer instance.
    rec.spawn().exit_on_failure();

    int time_sequence = 0;
    while (true) {
        sleep(1);
        time_sequence++;
    };
    return 0;
}
