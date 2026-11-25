# N-body -> hydro benchmark

- Input: `Mtot_Nbody`; target: hydro `Mtot` (IllustrisTNG), z = 0
- Weights: `none (baselines only)`
- Split seed 42: 800 train / 100 held-out test sims
- Spectra are of the density contrast, k in h/Mpc. P/P_true is the mean over maps.

## LH held-out test sims (1500 maps)

| Method | log-MSE (1e-3 dex²) | floored pixels | mass ratio | max \|P/P_true − 1\|, k ≤ 10 | P/P_true k=1 | P/P_true k=5 | P/P_true k=10 | P/P_true k=20 | r k=1 | r k=5 | r k=10 | r k=20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 9.101 | 0.00% | 0.9998 ± 0.0043 | 0.343 | 1.040 | 1.233 | 1.382 | 1.397 | 0.9995 | 0.9941 | 0.9725 | 0.9185 |
| amp_mean | 11.608 | 0.35% | 0.9998 ± 0.0043 | 0.033 | 0.998 | 1.021 | 1.039 | 1.056 | 0.9995 | 0.9941 | 0.9725 | 0.9185 |
| amp_emulated | 15.606 | 1.02% | 1.0001 ± 0.0044 | 0.004 | 1.000 | 1.004 | 1.006 | 1.013 | 0.9995 | 0.9941 | 0.9725 | 0.9184 |
| amp_oracle | 38.552 | 4.02% | 1.0022 ± 0.0057 | 0.005 | 0.995 | 0.995 | 0.995 | 0.995 | 0.9995 | 0.9942 | 0.9727 | 0.9187 |

## CV (fiducial model) (405 maps)

| Method | log-MSE (1e-3 dex²) | floored pixels | mass ratio | max \|P/P_true − 1\|, k ≤ 10 | P/P_true k=1 | P/P_true k=5 | P/P_true k=10 | P/P_true k=20 | r k=1 | r k=5 | r k=10 | r k=20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 8.015 | 0.00% | 0.9998 ± 0.0035 | 0.302 | 1.028 | 1.195 | 1.338 | 1.348 | 0.9997 | 0.9964 | 0.9810 | 0.9361 |
| amp_mean | 10.453 | 0.31% | 0.9999 ± 0.0035 | 0.018 | 0.987 | 0.989 | 1.007 | 1.019 | 0.9997 | 0.9965 | 0.9811 | 0.9360 |
| amp_emulated | 12.147 | 0.61% | 0.9999 ± 0.0035 | 0.008 | 0.993 | 1.000 | 1.005 | 1.008 | 0.9997 | 0.9965 | 0.9811 | 0.9360 |
| amp_oracle | 35.026 | 3.80% | 1.0018 ± 0.0043 | 0.005 | 0.995 | 0.996 | 0.996 | 0.996 | 0.9997 | 0.9965 | 0.9812 | 0.9363 |
