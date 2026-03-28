#include <ap_int.h>
#include <hls_stream.h>

// Definitions matching CNN and MLP
#define IN_CHANNELS 14
#define OUT_CHANNELS 64
#define IN_SIZE 8
#define IN_FEATURES 384
#define OUT_FEATURES 1536

typedef ap_fixed<16, 6> data_t;

// Structures matching cnn_layer.cpp
struct pixel_t {
    data_t ch[IN_CHANNELS];
};

struct out_pixel_t {
    data_t ch[OUT_CHANNELS];
};

// External function signatures matching actual definitions from sub-layers
void cnn_layer(
    hls::stream<pixel_t>& in_stream,
    hls::stream<out_pixel_t>& out_stream
);

void mlp_layer(
    hls::stream<data_t>& in_stream,
    data_t output_vec[OUT_FEATURES]
);

// --- Dataflow Adapter 1: Memory to Stream ---
// Converts memory-mapped array into a continuous pixel stream for the CNN
void read_input_map(
    data_t input_feature_map[IN_CHANNELS][IN_SIZE][IN_SIZE],
    hls::stream<pixel_t>& out_stream
) {
    for (int row = 0; row < IN_SIZE; row++) {
        for (int col = 0; col < IN_SIZE; col++) {
            #pragma HLS PIPELINE II=1
            pixel_t p;
            for (int ch = 0; ch < IN_CHANNELS; ch++) {
                p.ch[ch] = input_feature_map[ch][row][col];
            }
            out_stream.write(p);
        }
    }
}

// --- Dataflow Adapter 2: Stream Format Conversion & Drain ---
// Serializes CNN output structs into flat streaming data and drains excess values
void serialize_and_drain_cnn_output(
    hls::stream<out_pixel_t>& in_stream,
    hls::stream<data_t>& out_stream
) {
    int sent_count = 0;
    const int total_cnn_pixels = IN_SIZE * IN_SIZE;
    for (int i = 0; i < total_cnn_pixels; i++) {
        out_pixel_t p = in_stream.read();
        for (int ch = 0; ch < OUT_CHANNELS; ch++) {
            #pragma HLS PIPELINE II=1
            if (sent_count < IN_FEATURES) {
                out_stream.write(p.ch[ch]);
                sent_count++;
            }
        }
    }
}

// This is our TOP-LEVEL Tensor Processing Unit (TPU) Wrapper
// It connects the CNN Systolic Array directly into the MLP Matrix Engine!
void tensor_processing_unit(
    data_t input_feature_map[IN_CHANNELS][IN_SIZE][IN_SIZE],
    data_t mlp_output[OUT_FEATURES]
) {
    // Configure top-level AXI-Memory Mapped Ports for Processor/DDR Communication
    #pragma HLS INTERFACE m_axi port=input_feature_map offset=slave bundle=gmem0    // offset=slave to not hardcode pointer address,
    // and wait for the cpu to send data addressess; m_axi means FPGA is the master and has full control over the bus 
    #pragma HLS INTERFACE m_axi port=mlp_output offset=slave bundle=gmem1   // gemm1 is the bus name
    #pragma HLS INTERFACE s_axilite port=return bundle=control  // this interface is s_axi (slave) and waits for cpu (master) 
    // to send control signals like turn on or off the accelerator, and also to get the status of the accelerator

    // DATAFLOW PRAGMA: This forces Vitis HLS to synthesize both units 
    // simultaneously and run them concurrently on the FPGA!
    #pragma HLS DATAFLOW
    // Clock Cycle 1: The cnn_layer machine computes the very first pixel and drops it onto the hls::stream conveyor belt. 
    // It doesn't stop to wait; it immediately starts working on pixel #2.
    // Clock Cycle 2: The mlp_layer machine sees that pixel #1 has arrived on the belt. It picks it up and starts processing it. 
    // Meanwhile, the cnn_layer drops pixel #2 onto the belt/bus.
    // Clock Cycle 3... and beyond: Both the cnn_layer and mlp_layer are running entirely concurrently.

    // Internal inter-module AXI-Streams
    hls::stream<pixel_t> input_stream("input_stream");
    hls::stream<out_pixel_t> cnn_to_adapter_stream("cnn_out_stream");
    hls::stream<data_t> adapter_to_mlp_stream("mlp_in_stream");

    // Launch Block 0: Memory Read Adapter
    read_input_map(input_feature_map, input_stream);

    // Launch Block 1: Conv2D
    // Outputs 64-channel out_pixel_t format
    cnn_layer(input_stream, cnn_to_adapter_stream);

    // Launch Block 2: Serializer and Drain Adapter
    // Converts out_pixel_t to data_t stream and prevents deadlocks by draining remainder
    serialize_and_drain_cnn_output(cnn_to_adapter_stream, adapter_to_mlp_stream);

    // Launch Block 3: Transformer MLP
    // Reads exactly 384 elements representing one embedded token
    mlp_layer(adapter_to_mlp_stream, mlp_output);
}
