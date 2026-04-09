#!/usr/bin/env python3
"""
fpga_estimation_zcu102.py

Analytical FPGA resource projections for AeroConv1D on Xilinx ZCU102.
Produces the values reported in Table IV of the paper.

Scaling model (hls4ml-derived, parameter-proportional):
  LUT  = n_params * bits / 6
         Each parameter stored as a bits-wide word; Xilinx LUT6 can hold
         6 bits of configuration, giving a linear LUT cost per parameter-bit.
  DSP  = n_params * bits / 18
         DSP48E2 natively implements 18-bit operands. A b-bit fixed-point
         weight therefore requires ceil(b/18) DSP slices; aggregated over
         all parameters: DSP = n_params * bits / 18.
  Lat  = bits / 2  µs
         At 500 MHz, the pipelined inference latency scales with bit-width
         (more bits -> deeper adder tree). FP32 achieves 16 us; lower
         precisions scale proportionally: lat_us = bits / 2.
  Comm = n_params * bits / 8 / 1024  KiB/round
         Raw gradient payload; excludes per-layer scale overhead
         (8 layers x 4 B ~= 0.03 KiB, negligible).

All values are ANALYTICAL PROJECTIONS not validated on physical silicon.

Run: python fpga_estimation_zcu102.py
"""

import json
import os

os.makedirs("results", exist_ok=True)

# -- ZCU102 resources ----------------------------------------------------------
ZCU102 = {
    "part"   : "xczu9eg-ffvb1156-2-e",
    "LUT"    : 274_080,
    "FF"     : 548_160,
    "DSP"    : 2_520,
    "BRAM36" : 912,
}

# -- AeroConv1D parameter count (verified programmatically) -------------------
LAYER_PARAMS = {
    "Conv1D(14->32, k=3)": 14 * 32 * 3 + 32,   # weight + bias = 1,376
    "Conv1D(32->64, k=3)": 32 * 64 * 3 + 64,   # weight + bias = 6,208
    "Linear(64->32)"     : 64 * 32     + 32,   # weight + bias = 2,080
    "Linear(32->1)"      : 32 * 1      + 1,    # weight + bias =    33
}
N_PARAMS = sum(LAYER_PARAMS.values())
assert N_PARAMS == 9_697, f"Parameter count mismatch: {N_PARAMS}"

# -- Precision configurations --------------------------------------------------
CONFIGS = {
    "FP32": {"bits": 32, "precision": "ap_fixed<32,16>"},
    "INT8": {"bits":  8, "precision": "ap_fixed<8,4>"},
    "INT4": {"bits":  4, "precision": "ap_fixed<4,2>"},
    "INT2": {"bits":  2, "precision": "ap_fixed<2,1>"},
}

# -- Estimation ----------------------------------------------------------------
def estimate(bits):
    lut       = int(N_PARAMS * bits / 6)
    dsp       = int(N_PARAMS * bits / 18)
    lat_us    = bits // 2          # 32->16, 8->4, 4->2, 2->1 us
    comm_kib  = round(N_PARAMS * bits / 8 / 1024, 2)
    pct_lut   = round(lut / ZCU102["LUT"] * 100, 1)
    pct_dsp   = round(dsp / ZCU102["DSP"] * 100, 1)
    fits      = pct_dsp <= 100.0 and pct_lut <= 100.0
    spare_dsp = max(0, ZCU102["DSP"] - dsp)
    return {
        "lut": lut, "pct_lut": pct_lut,
        "dsp": dsp, "pct_dsp": pct_dsp,
        "latency_us": lat_us,
        "comm_kib": comm_kib,
        "fits": fits,
        "spare_dsps": spare_dsp,
    }

# -- Display -------------------------------------------------------------------
print(f"AeroConv1D: {N_PARAMS:,} parameters")
for k, v in LAYER_PARAMS.items():
    print(f"  {k}: {v:,}")
print()

sep = "=" * 76
print(sep)
print(f"{'Config':<6} {'Precision':<18} {'LUT':>8} {'%LUT':>6} "
      f"{'DSP':>7} {'%DSP':>7} {'Lat':>6} {'Comm KiB':>10}  Fit")
print("-" * 76)

all_results = {}
for label, cfg in CONFIGS.items():
    r = estimate(cfg["bits"])
    all_results[label] = {**cfg, **r}
    fit_str = "OK" if r["fits"] else "X"
    print(f"{label:<6} {cfg['precision']:<18} "
          f"{r['lut']:>8,} {r['pct_lut']:>5.1f}% "
          f"{r['dsp']:>7,} {r['pct_dsp']:>6.1f}% "
          f"{r['latency_us']:>4} us "
          f"{r['comm_kib']:>10.2f}  {fit_str}")

print(sep)
print(f"\nZCU102 budget: {ZCU102['LUT']:,} LUT | "
      f"{ZCU102['DSP']:,} DSP | {ZCU102['BRAM36']} BRAM36")

# -- Verdict -------------------------------------------------------------------
print("\nDeployment verdict:")
for label, r in all_results.items():
    verdict = "DEPLOYABLE" if r["fits"] else "EXCEEDS BUDGET"
    spare   = f" ({r['spare_dsps']} spare DSPs)" if r["fits"] else ""
    print(f"  {label:<5}: LUT {r['pct_lut']:5.1f}%  "
          f"DSP {r['pct_dsp']:5.1f}%  -> {verdict}{spare}")

int4 = all_results["INT4"]
print(f"\nINT4 key figures (Table IV of the paper):")
print(f"  DSP utilisation : {int4['pct_dsp']:.1f}%")
print(f"  Spare DSPs      : {int4['spare_dsps']}"
      f"  (headroom for NTT-based HE co-processor)")
print(f"  Latency         : {int4['latency_us']} us at 500 MHz")
print(f"  Comm. cost      : {int4['comm_kib']:.2f} KiB/round")

# -- Save ----------------------------------------------------------------------
for label, r in all_results.items():
    with open(f"results/fpga_zcu102_{label}.json", "w") as f:
        json.dump({**CONFIGS[label], **r, "part": ZCU102["part"]}, f, indent=2)

with open("results/fpga_zcu102_all.json", "w") as f:
    json.dump({
        "device"     : ZCU102,
        "model"      : {"n_params": N_PARAMS, "layer_params": LAYER_PARAMS},
        "methodology": (
            "LUT = n_params*bits/6 | DSP = n_params*bits/18 | "
            "Latency = bits/2 us at 500 MHz | Comm = n_params*bits/8/1024 KiB"
        ),
        "configs"    : all_results,
    }, f, indent=2)

print("\nResults saved to results/fpga_zcu102_*.json")
print("Note: All figures are ANALYTICAL PROJECTIONS.")
print("      Not validated on physical ZCU102 silicon.")
