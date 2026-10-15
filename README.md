# PDSCH Link Simulation

A Python simulation of a single-antenna PDSCH link. It follows a transport block through encoding, modulation, OFDM, the channel, and decoding, then checks the recovered bits with CRC. The project also includes BLER sweeps and plots of BER, EVM, and throughput for AWGN and TDL-C channels.

For a first look at the code, start with `pdsch_e2e.py`. It connects the individual modules and defines the link parameters in `LinkConfig`. `bler_curve.py` runs repeated trials over a range of SNR values.

## Run the link

From the project directory, install the dependencies and run the end-to-end check. The project was tested with Python 3.14.2 and the package versions in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python pdsch_e2e.py
```

Use `python3` instead of `python` if needed on your system.

This checks AWGN and TDL-C cases with ideal and estimated channel information. It verifies the decoded bits and CRC, and prints `pdsch e2e checks passed` when all cases pass.

To run the default SNR sweeps and generate plots:

```sh
python bler_curve.py
```

The script creates `results/` for CSV files and `figures/` for PNG plots. These are generated outputs and are not tracked in the repository. The default run produces five CSV files and four figures. It takes longer than the self-checks because it simulates multiple blocks at each SNR point.

Other options for `bler_curve.py`:

| Option | What it does |
| --- | --- |
| `--no-plot` | Run the simulation and save CSV files without generating plots |
| `--full` | Use more blocks per SNR point |
| `--phy` | Compare uncoded QPSK BER with the AWGN theoretical curve |
| `--replot` | Regenerate plots from existing CSV files; run a sweep first |

## Code layout

The transmit path is:

```text
Transport block -> CRC-24A -> LDPC -> Rate matching -> Scrambling
                -> QAM -> Resource mapping with DM-RS -> OFDM
```

The receiver reverses this process, using channel estimation and equalization before soft demodulation. It then descrambles the LLRs, recovers the rate-matched soft information, decodes the LDPC codeword, and checks the CRC.

| File | Role |
| --- | --- |
| `pdsch_e2e.py` | Link configuration, transmit/receive chain, and slot simulation |
| `bler_curve.py` | SNR sweeps, error statistics, throughput, and plotting |
| `CRC24A.py`, `gold_sequence.py` | CRC and Gold sequence generation |
| `ldpc_encoder.py`, `ldpc_decoder.py` | LDPC encoding and Min-Sum / Sum-Product decoding |
| `rate_matching.py` | Rate matching and LLR recovery |
| `qam_constellation.py`, `modulatordemodulator.py` | QPSK through 256QAM constellations and soft demodulation |
| `dmrs.py`, `re_mapping.py` | DM-RS generation and resource mapping |
| `ofdm.py`, `channel_equ.py` | OFDM, channel models, channel estimation, and ZF / MMSE equalization |

## Module checks

The checks use Python's standard library and the project dependencies; no separate test framework is needed. Run them from the project directory:

```sh
python test_gold.py
python test_ofdm.py
python testmod.py
python crctest.py
python qam_constellation.py
python dmrs.py
python re_mapping.py
python channel_equ.py
python ldpc_encoder.py
python rate_matching.py
python ldpc_decoder.py
```

These cover sequence properties, CRC examples, constellation mappings, modulation and demodulation, OFDM round trips, resource mapping, channel equalization, both LDPC base graphs, and rate matching.

The module checks, end-to-end check, BLER self-check, and default sweep all passed in a separate directory containing only the files intended for this repository, using the Python environment listed above.

## Data files

Keep these files in the same directory as the Python scripts:

- `shift_coeffs_bg_1.mat` and `shift_coeffs_bg_2.mat`: LDPC base-graph shift coefficients used by the encoder and decoder.
- `qam_ref_38_211.csv`: Reference constellation values used by the QAM checks.

SciPy reads the MAT files directly. MATLAB and Excel are not needed to run the project.

## Current scope

The default link uses SISO, 30 kHz subcarrier spacing, a 1024-point FFT, 20 PRBs, 14 OFDM symbols per slot, QPSK, and a 76-bit transport block. The small transport block keeps the initial checks quick. The BLER script includes presets with different PRB counts and transport block sizes.

The implementation handles a single code block per transport block. Transport-block segmentation, multi-block reassembly, and a full HARQ protocol are not implemented. It is an algorithm-level simulation and has not been through a full standards conformance test.

