#include <iostream>
#include <ap_int.h>
#include <cstdlib>

// Definitions matching the tensor_processing_unit
#define IN_CHANNELS 14
#define OUT_CHANNELS 64
#define IN_SIZE 8
#define IN_FEATURES 384
#define OUT_FEATURES 1536

typedef ap_fixed<16, 6> data_t;

// Function prototype for the top-level TPU
void tensor_processing_unit(
    data_t input_feature_map[IN_CHANNELS][IN_SIZE][IN_SIZE],
    data_t mlp_output[OUT_FEATURES]
);

int main() {
    // 1. Allocate memory for inputs and outputs
    static data_t input_feature_map[IN_CHANNELS][IN_SIZE][IN_SIZE];
    static data_t mlp_output[OUT_FEATURES];

    // 2. Initialize inputs with test values
    std::cout << "============================================" << std::endl;
    std::cout << "Starting Testbench for Tensor Processing Unit" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Initializing inputs..." << std::endl;
    for (int c = 0; c < IN_CHANNELS; c++) {
        for (int h = 0; h < IN_SIZE; h++) {
            for (int w = 0; w < IN_SIZE; w++) {
                // Initialize with some arbitrary small small values to prevent overflow
                // e.g., using modulo to keep values small and predictable
                input_feature_map[c][h][w] = (data_t)(((c + h + w) % 5) * 0.1); 
            }
        }
    }

    // 3. Clear the output array to known state
    for (int i = 0; i < OUT_FEATURES; i++) {
        mlp_output[i] = 0;
    }

    // 4. Call the hardware function (Top-Level)
    std::cout << "Calling tensor_processing_unit()..." << std::endl;
    tensor_processing_unit(input_feature_map, mlp_output);
    std::cout << "Execution completed." << std::endl;

    // 5. Verify the results (Print a few sample outputs)
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Sample Outputs:" << std::endl;
    for (int i = 0; i < 15; i++) {
        std::cout << "mlp_output[" << i << "] = " << (float)mlp_output[i] << std::endl;
    }
    std::cout << "..." << std::endl;
    for (int i = OUT_FEATURES - 5; i < OUT_FEATURES; i++) {
        std::cout << "mlp_output[" << i << "] = " << (float)mlp_output[i] << std::endl;
    }
    std::cout << "--------------------------------------------" << std::endl;

    // 6. Return 0 to indicate successful simulation
    std::cout << "Testbench finished successfully!" << std::endl;
    return 0;
}
