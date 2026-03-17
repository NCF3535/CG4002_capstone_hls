# CG4002 Capstone HLS - PickleballNet

Vitis HLS implementation of the PickleballNet accelerator for the Ultra96-V2 (Zynq UltraScale+ ZU3EG). Part of the CG4002 capstone AI system — the training pipeline and deployment code are maintained in a separate repository:
https://github.com/CG4002-AY2526S2-B03/CG4002_capstone_AI_Accelerator

---

## Overview

PickleballNet is a multi-task neural network that takes a 6D ball state (position + velocity) as input and simultaneously predicts:
- **Regression:** 6D racket target state (position + velocity)
- **Classification:** Shot type (Drive, Drop, Dink, Lob, SpeedUp, HandBattle)

The trained weights are quantized to INT8, exported as C headers, and synthesized with Vitis HLS into an AXI-Stream IP core integrated via a Vivado block design.

**Architecture:** Shared FC trunk (2 x 512, BatchNorm fused, ReLU6) -> RegHead (512->256->6) + ClsHead (512->256->6)

**Interface:** AXI-Stream — 6 x float32 in, 12 x float32 out (6 regression + 6 classification logits)

---

## Repository Structure

```
pickleball_model.cpp   - HLS top-level function (pb_predict)
pickleball_model.h     - Interface and dimension defines
weights.h              - INT8 quantized weights (exported from training repo)
tb_pickleball.cpp      - C simulation testbench
test_vectors.h         - Test input/output vectors (exported from training repo)
pickleball_hls/        - Vitis HLS project
Pickleball_vivado/     - Vivado block design project
```

---

## Resource Targets (ZU3EG)

| Resource | Available | Target |
|---|---|---|
| LUT | 70,560 | - |
| BRAM36K | 216 | ~128 (59%) |
| DSP | 360 | - |

---

## Pipeline Notes

- **LAYER0** (6->512): outer loop pipelined II=1, inner loop fully unrolled. 6 parallel fmuls + adder tree, no loop-carried dependency.
- **LAYER1 + heads** (512->512, 512->256): inner loop pipelined II=4 using 16 independent partial accumulators to break false resource conflicts.

---

## Memory Strategy

| Storage | Contents |
|---|---|
| BRAM (rom_2p, cyclic=16) | trunk_1, reg_head_0, cls_head_0 weights |
| LUTRAM (rom_1p) | trunk_0, reg_head_1, cls_head_1 weights, all biases, scalers |
| FFs (cyclic=16 partition) | h_a, h_b, head_buf activation buffers |
| FFs (complete partition) | input_raw, input_scaled, reg_out, cls_out |

---

## Build

**Step 1 — HLS synthesis:** Open `pickleball_hls/hls.app` in Vitis HLS and run C Synthesis + Export RTL.

**Step 2 — Vivado implementation:** Open `Pickleball_vivado/Pickleball_vivado.xpr` in Vivado 2022.1, run synthesis, implementation, and generate bitstream.

The generated `design_1.bit` and `design_1.hwh` are then used by the deployment scripts in the [AI Accelerator repo](https://github.com/CG4002-AY2526S2-B03/CG4002_capstone_AI_Accelerator) (`ultra96_deploy/`).

**Block design:** `Pickleball_vivado/Pickleball_vivado.srcs/sources_1/bd/design_1/design_1.bd`
