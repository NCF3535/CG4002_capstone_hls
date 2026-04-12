



















#include "pickleball_model.h"
#include "weights.h"
#include <hls_math.h>

union float_uint32 {
    float f;
    unsigned int u;
};

static inline float relu6(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 6.0f) return 6.0f;
    return x;
}




static void run_reg_head(float h_b[HIDDEN], float reg_out[OUT_REG]) {
    #pragma HLS ARRAY_PARTITION variable=h_b cyclic factor=16
    float reg_buf[HEAD_HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=reg_buf cyclic factor=16

    
    REG_HEAD_L0:
    for (int i = 0; i < HEAD_HIDDEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        REG_H0_MAC:
        for (int j = 0; j < HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 0] * h_b[j+ 0];
            s1  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 1] * h_b[j+ 1];
            s2  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 2] * h_b[j+ 2];
            s3  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 3] * h_b[j+ 3];
            s4  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 4] * h_b[j+ 4];
            s5  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 5] * h_b[j+ 5];
            s6  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 6] * h_b[j+ 6];
            s7  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 7] * h_b[j+ 7];
            s8  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 8] * h_b[j+ 8];
            s9  += (float)reg_head_0_weight_q[i*HIDDEN+j+ 9] * h_b[j+ 9];
            s10 += (float)reg_head_0_weight_q[i*HIDDEN+j+10] * h_b[j+10];
            s11 += (float)reg_head_0_weight_q[i*HIDDEN+j+11] * h_b[j+11];
            s12 += (float)reg_head_0_weight_q[i*HIDDEN+j+12] * h_b[j+12];
            s13 += (float)reg_head_0_weight_q[i*HIDDEN+j+13] * h_b[j+13];
            s14 += (float)reg_head_0_weight_q[i*HIDDEN+j+14] * h_b[j+14];
            s15 += (float)reg_head_0_weight_q[i*HIDDEN+j+15] * h_b[j+15];
        }
        reg_buf[i] = relu6((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * reg_head_0_qscale + reg_head_0_bias[i]);
    }

    
    REG_HEAD_L1:
    for (int i = 0; i < OUT_REG; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        REG_H1_MAC:
        for (int j = 0; j < HEAD_HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 0] * reg_buf[j+ 0];
            s1  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 1] * reg_buf[j+ 1];
            s2  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 2] * reg_buf[j+ 2];
            s3  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 3] * reg_buf[j+ 3];
            s4  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 4] * reg_buf[j+ 4];
            s5  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 5] * reg_buf[j+ 5];
            s6  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 6] * reg_buf[j+ 6];
            s7  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 7] * reg_buf[j+ 7];
            s8  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 8] * reg_buf[j+ 8];
            s9  += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+ 9] * reg_buf[j+ 9];
            s10 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+10] * reg_buf[j+10];
            s11 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+11] * reg_buf[j+11];
            s12 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+12] * reg_buf[j+12];
            s13 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+13] * reg_buf[j+13];
            s14 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+14] * reg_buf[j+14];
            s15 += (float)reg_head_1_weight_q[i*HEAD_HIDDEN+j+15] * reg_buf[j+15];
        }
        float y_norm = (s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * reg_head_1_qscale + reg_head_1_bias[i];
        reg_out[i] = y_norm * y_scaler_scale[i] + y_scaler_mean[i];
    }
}

static void run_cls_head(float h_b[HIDDEN], float cls_out[OUT_CLS]) {
    #pragma HLS ARRAY_PARTITION variable=h_b cyclic factor=16
    float cls_buf[HEAD_HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=cls_buf cyclic factor=16

    
    CLS_HEAD_L0:
    for (int i = 0; i < HEAD_HIDDEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        CLS_H0_MAC:
        for (int j = 0; j < HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 0] * h_b[j+ 0];
            s1  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 1] * h_b[j+ 1];
            s2  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 2] * h_b[j+ 2];
            s3  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 3] * h_b[j+ 3];
            s4  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 4] * h_b[j+ 4];
            s5  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 5] * h_b[j+ 5];
            s6  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 6] * h_b[j+ 6];
            s7  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 7] * h_b[j+ 7];
            s8  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 8] * h_b[j+ 8];
            s9  += (float)cls_head_0_weight_q[i*HIDDEN+j+ 9] * h_b[j+ 9];
            s10 += (float)cls_head_0_weight_q[i*HIDDEN+j+10] * h_b[j+10];
            s11 += (float)cls_head_0_weight_q[i*HIDDEN+j+11] * h_b[j+11];
            s12 += (float)cls_head_0_weight_q[i*HIDDEN+j+12] * h_b[j+12];
            s13 += (float)cls_head_0_weight_q[i*HIDDEN+j+13] * h_b[j+13];
            s14 += (float)cls_head_0_weight_q[i*HIDDEN+j+14] * h_b[j+14];
            s15 += (float)cls_head_0_weight_q[i*HIDDEN+j+15] * h_b[j+15];
        }
        cls_buf[i] = relu6((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * cls_head_0_qscale + cls_head_0_bias[i]);
    }

    
    CLS_HEAD_L1:
    for (int i = 0; i < OUT_CLS; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        CLS_H1_MAC:
        for (int j = 0; j < HEAD_HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 0] * cls_buf[j+ 0];
            s1  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 1] * cls_buf[j+ 1];
            s2  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 2] * cls_buf[j+ 2];
            s3  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 3] * cls_buf[j+ 3];
            s4  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 4] * cls_buf[j+ 4];
            s5  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 5] * cls_buf[j+ 5];
            s6  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 6] * cls_buf[j+ 6];
            s7  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 7] * cls_buf[j+ 7];
            s8  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 8] * cls_buf[j+ 8];
            s9  += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+ 9] * cls_buf[j+ 9];
            s10 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+10] * cls_buf[j+10];
            s11 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+11] * cls_buf[j+11];
            s12 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+12] * cls_buf[j+12];
            s13 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+13] * cls_buf[j+13];
            s14 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+14] * cls_buf[j+14];
            s15 += (float)cls_head_1_weight_q[i*HEAD_HIDDEN+j+15] * cls_buf[j+15];
        }
        cls_out[i] = (s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * cls_head_1_qscale + cls_head_1_bias[i];
    }
}


static void run_heads(
    float h_b[HIDDEN],
    float reg_out[OUT_REG],
    float cls_out[OUT_CLS]
) {
    #pragma HLS DATAFLOW
    run_reg_head(h_b, reg_out);
    run_cls_head(h_b, cls_out);
}

void pb_predict(
    hls::stream<axis_pkt_t> &input_stream,
    hls::stream<axis_pkt_t> &output_stream
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    
    float input_raw[IN_DIM];
    float input_scaled[IN_DIM];
    float reg_out[OUT_REG];
    float cls_out[OUT_CLS];
    #pragma HLS ARRAY_PARTITION variable=input_raw    complete
    #pragma HLS ARRAY_PARTITION variable=input_scaled complete
    #pragma HLS ARRAY_PARTITION variable=reg_out      complete
    #pragma HLS ARRAY_PARTITION variable=cls_out      complete

    
    
    float h_a[HIDDEN];
    float h_b[HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=h_a cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=h_b cyclic factor=16

    
    
    #pragma HLS BIND_STORAGE variable=trunk_1_weight_q    type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=reg_head_0_weight_q type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=cls_head_0_weight_q type=rom_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=trunk_1_weight_q    cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=reg_head_0_weight_q cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=cls_head_0_weight_q cyclic factor=16

    
    #pragma HLS BIND_STORAGE variable=trunk_0_weight_q    type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=reg_head_1_weight_q type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=cls_head_1_weight_q type=rom_1p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=trunk_0_weight_q    cyclic factor=6
    #pragma HLS ARRAY_PARTITION variable=reg_head_1_weight_q cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=cls_head_1_weight_q cyclic factor=16

    #pragma HLS BIND_STORAGE variable=trunk_0_bias    type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=trunk_1_bias    type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=reg_head_0_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=reg_head_1_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=cls_head_0_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=cls_head_1_bias type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=x_scaler_mean   type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=x_scaler_scale  type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=y_scaler_mean   type=rom_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=y_scaler_scale  type=rom_1p impl=lutram

    
    READ_INPUT:
    for (int i = 0; i < IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        axis_pkt_t pkt = input_stream.read();
        float_uint32 conv;
        conv.u = pkt.data;
        input_raw[i] = conv.f;
    }

    
    SCALE_INPUT:
    for (int i = 0; i < IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        input_scaled[i] = (input_raw[i] - x_scaler_mean[i]) / x_scaler_scale[i];
    }

    
    
    
    
    LAYER0:
    for (int i = 0; i < HIDDEN; i++) {
        #pragma HLS PIPELINE
        float sum = 0.0f;
        L0_MAC:
        for (int j = 0; j < IN_DIM; j++) {
            #pragma HLS UNROLL
            sum += (float)trunk_0_weight_q[i * IN_DIM + j] * input_scaled[j];
        }
        h_a[i] = relu6(sum * trunk_0_qscale + trunk_0_bias[i]);
    }

    
    
    
    
    LAYER1:
    for (int i = 0; i < HIDDEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
        float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
        L1_MAC:
        for (int j = 0; j < HIDDEN; j += 16) {
            #pragma HLS PIPELINE II=4
            s0  += (float)trunk_1_weight_q[i*HIDDEN+j+ 0] * h_a[j+ 0];
            s1  += (float)trunk_1_weight_q[i*HIDDEN+j+ 1] * h_a[j+ 1];
            s2  += (float)trunk_1_weight_q[i*HIDDEN+j+ 2] * h_a[j+ 2];
            s3  += (float)trunk_1_weight_q[i*HIDDEN+j+ 3] * h_a[j+ 3];
            s4  += (float)trunk_1_weight_q[i*HIDDEN+j+ 4] * h_a[j+ 4];
            s5  += (float)trunk_1_weight_q[i*HIDDEN+j+ 5] * h_a[j+ 5];
            s6  += (float)trunk_1_weight_q[i*HIDDEN+j+ 6] * h_a[j+ 6];
            s7  += (float)trunk_1_weight_q[i*HIDDEN+j+ 7] * h_a[j+ 7];
            s8  += (float)trunk_1_weight_q[i*HIDDEN+j+ 8] * h_a[j+ 8];
            s9  += (float)trunk_1_weight_q[i*HIDDEN+j+ 9] * h_a[j+ 9];
            s10 += (float)trunk_1_weight_q[i*HIDDEN+j+10] * h_a[j+10];
            s11 += (float)trunk_1_weight_q[i*HIDDEN+j+11] * h_a[j+11];
            s12 += (float)trunk_1_weight_q[i*HIDDEN+j+12] * h_a[j+12];
            s13 += (float)trunk_1_weight_q[i*HIDDEN+j+13] * h_a[j+13];
            s14 += (float)trunk_1_weight_q[i*HIDDEN+j+14] * h_a[j+14];
            s15 += (float)trunk_1_weight_q[i*HIDDEN+j+15] * h_a[j+15];
        }
        h_b[i] = relu6((s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15) * trunk_1_qscale + trunk_1_bias[i]);
    }

    
    run_heads(h_b, reg_out, cls_out);

    
    WRITE_REG:
    for (int i = 0; i < OUT_REG; i++) {
        #pragma HLS PIPELINE II=1
        axis_pkt_t pkt;
        float_uint32 conv;
        conv.f = reg_out[i];
        pkt.data = conv.u;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = 0;
        output_stream.write(pkt);
    }

    WRITE_CLS:
    for (int i = 0; i < OUT_CLS; i++) {
        #pragma HLS PIPELINE II=1
        axis_pkt_t pkt;
        float_uint32 conv;
        conv.f = cls_out[i];
        pkt.data = conv.u;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = (i == OUT_CLS - 1) ? 1 : 0;
        output_stream.write(pkt);
    }
}
