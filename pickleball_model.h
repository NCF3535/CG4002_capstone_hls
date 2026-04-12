#ifndef PICKLEBALL_MODEL_H
#define PICKLEBALL_MODEL_H

#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <ap_fixed.h>









#define IN_DIM        6
#define HIDDEN        512
#define HEAD_HIDDEN   256
#define N_LAYERS      2
#define OUT_REG       6
#define OUT_CLS       6
#define OUT_TOTAL     (OUT_REG + OUT_CLS)
#define UNROLL_FACTOR 4

typedef ap_axis<32, 0, 0, 0> axis_pkt_t;

void pb_predict(
    hls::stream<axis_pkt_t> &input_stream,
    hls::stream<axis_pkt_t> &output_stream
);

#endif 
