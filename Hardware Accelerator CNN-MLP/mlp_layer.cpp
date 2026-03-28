#include <ap_int.h>
#include <hls_stream.h>
#include "weights.h"

// Example: mlp_fc1 from Transformer Block (n_embd=384 -> 4*n_embd=1536)
#define IN_FEATURES 384
#define OUT_FEATURES 1536

// Define hardware factors for parallelism (similar to CNN channel looping)
#define BLOCK_SIZE 16   // Array partitioning and loop unrolling factor

typedef ap_fixed<16, 6> data_t; // fixed-point datatype, quantization to 16bit

// Accelerating MLP using AXI Stream Inputs and Block-Level Matrix Multiplication
void mlp_layer(
    hls::stream<data_t>& in_stream,
    data_t output_vec[OUT_FEATURES]
) {
    // --- 1. HARDWARE REGISTERS & BUFFERS ---
    // Array to hold the incoming stream for reuse across matrix rows
    // 'static' ensures this memory holds its values between function calls if needed
    static data_t local_input_buffer[IN_FEATURES];
    
    // Partition the input buffer cyclically so Vivado can read BLOCK_SIZE elements in parallel!
    #pragma HLS ARRAY_PARTITION variable=local_input_buffer cyclic factor=16 dim=1
    
    // Partition weights along the inner dimension to match the input block size
    #pragma HLS ARRAY_PARTITION variable=mlp_weights cyclic factor=16 dim=2

    // --- 2. STREAM INPUT ---
    // Read the AXI stream once and store in ultra-fast distributed RAM (BRAM)
    for(int j = 0; j < IN_FEATURES; j++) {
        #pragma HLS PIPELINE II=1
        local_input_buffer[j] = in_stream.read();
    }

    // --- 3. DENSE MULTIPLY-ACCUMULATE (MAC) ENGINE ---
    for (int i = 0; i < OUT_FEATURES; i++) {
        data_t acc = mlp_bias[i];
        
        // Tiled Matrix Multiplication (Block Processing)
        for (int j = 0; j < IN_FEATURES; j += BLOCK_SIZE) {
            #pragma HLS PIPELINE II=1 // Instructs Vivado to pipeline this loop (throughput = 1 result per cycle)
            
            data_t sum = 0;
            
            // This inner loop will be fully unrolled into parallel DSP multipliers (16 multipliers per cycle)
            for (int k = 0; k < BLOCK_SIZE; k++) {
                #pragma HLS UNROLL
                sum += local_input_buffer[j + k] * mlp_weights[i][j + k];
            }
            
            acc += sum; // Accumulate the block sum
        }
        
        // --- 4. RELU ACTIVATION ---
        if (acc < 0) acc = 0; // Hardware logic gate for ReLU
        
        // Write the result to memory
        output_vec[i] = acc;
    }
}
