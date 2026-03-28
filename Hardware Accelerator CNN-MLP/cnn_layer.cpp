#include <ap_int.h>     // Vitis High-Level Synthesis library for datatypes
#include <hls_stream.h> // library for FIFO stream constructs
#include "weights.h"    // pre-trained weights

// Define MACROS: since appl. specific so no dynamic memory allocation at compile time, instead exact sizes defined to wire logic gates
#define IN_CHANNELS 14       // 14 Chess Board State Masks
#define OUT_CHANNELS 64      // First Conv2d output depth
#define IN_SIZE 8            // 8x8 Chess Board
#define KERNEL_SIZE 3        // 3x3 Sliding Window Filter
#define PAD 1
#define PADDED_SIZE (IN_SIZE + 2 * PAD) // 8 + 2(1) = 10

typedef ap_fixed<16, 6> data_t; // fixed-point datatype, quantization to 16bit, where 6bits are exponent

struct pixel_t {
    data_t ch[IN_CHANNELS];     // Bundles all input channels (14) into a single pixel_t struct
};

struct out_pixel_t {
    data_t ch[OUT_CHANNELS];    // Bundles all output channels (64) into a single out_pixel_t struct
};

// Accelerating CNN using Systolic Arrays and AXI Stream
void cnn_layer(
    hls::stream<pixel_t>& in_stream,    // synthesize into AXI4-Stream hardware ports (FIFO queues) rather than standard memory addresses.
    hls::stream<out_pixel_t>& out_stream
) {
    // --- 1. HARDWARE REGISTERS ---
    // allocate local memory to line buffer. static to ensure this memory holds its values between function calls (acting as persistent hardware state
    // image data arrives as a stream of pixels (row by row). A line buffer stores the previous rows.
    static data_t line_buffer[IN_CHANNELS][KERNEL_SIZE - 1][PADDED_SIZE];   // 14 channels, 2 rows, 10 columns
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
    
    static data_t window[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // --- 2. PADDED SPATIAL LOOPS ---
    // Instead of 8x8, the hardware loops over the 10x10 padded grid
    for (int row = 0; row < PADDED_SIZE; row++) {
        for (int col = 0; col < PADDED_SIZE; col++) {
            
            #pragma HLS PIPELINE II=1   // placed inside the col loop. Because of this, 
            // the compiler will automatically try to unroll all loops beneath this point so they execute in parallel.
            
            // --- 3. ZERO-INJECTION PADDING LOGIC ---
            pixel_t new_pixel;  // operations happen one pixel at a time (raster-scan order) but simultaneously for all 64 output channels
            
            // If we are inside the actual 8x8 image boundaries, read from stream.
            // Otherwise, inject hardware zeros.
            if (row >= PAD && row < IN_SIZE + PAD && col >= PAD && col < IN_SIZE + PAD) {
                new_pixel = in_stream.read(); // Read valid pixel
            } else {
                for (int ic = 0; ic < IN_CHANNELS; ic++) {
                    new_pixel.ch[ic] = 0; // Inject Zero Padding
                }
            }
            
            // --- 4. SHIFT LOGIC  ---
            for (int ic = 0; ic < IN_CHANNELS; ic++) {
                // Shift window left
                for (int i = 0; i < KERNEL_SIZE; i++) {
                    for (int j = 0; j < KERNEL_SIZE - 1; j++) {
                        window[ic][i][j] = window[ic][i][j + 1];    // The Left column [0] gets overwritten by the Middle column [1] & so on...
                    }                                               // rightmost column [2] is empty
                }
                
                // Shift line buffer up and into window
                window[ic][0][KERNEL_SIZE - 1] = line_buffer[ic][0][col]; // Feed the [0][2] of the window from the oldest line buffer (Row 0)
                window[ic][1][KERNEL_SIZE - 1] = line_buffer[ic][1][col]; // Feed the [1][2] of the window from the second oldest line buffer (Row 1)
                line_buffer[ic][0][col] = line_buffer[ic][1][col];    // The pixel currently in line buffer row 1 gets pushed up to row 0
                
                window[ic][2][KERNEL_SIZE - 1] = new_pixel.ch[ic];  // Feed the bottom-right from the brand new pixel!
                line_buffer[ic][1][col] = new_pixel.ch[ic];         // The brand new pixel gets saved into line buffer row 1
            }

            // --- 5. THE MAC ENGINE ---
            if (row >= KERNEL_SIZE - 1 && col >= KERNEL_SIZE - 1) { // wait until the window is full i.e row and col are 2
                
                out_pixel_t output_data;
                
                for (int oc = 0; oc < OUT_CHANNELS; oc++) {
                    data_t acc = conv_bias[oc];     // initialize accumulator with layer's bias rather than at the end for efficiency
                    
                    for (int ic = 0; ic < IN_CHANNELS; ic++) {
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                                acc += window[ic][kr][kc] * conv_weights[oc][ic][kr][kc];
                            }
                        }
                    }
                    
                    if (acc < 0) acc = 0;   // ReLU Activation in Hardware
                    output_data.ch[oc] = acc;
                }
                
                out_stream.write(output_data);  // Push immediately to the AXI Stream!
            }
        }
    }
}
