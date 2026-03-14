# SyntheticControlMethods reviewer-response supplement

This folder contains the code files needed for the Figure 2, Figure 3, and reviewer-response simulations discussed in the JCI revision.

## Files

- `mmscm.py`
  - core MMSCM implementation
  - includes the additional arguments used in the reviewer-response experiment, following the uploaded `mmscm.py`
- `Fig02_Fig03_Simulation_Estimation_Error_and_ATE_MSE.ipynb`
  - simulation notebook for Figure 2 and Figure 3
  - includes the SCPI point-estimator comparison already added to the existing simulation loop
- `Fig02_Fig03_Reviewer_Response_hG_and_Loss.ipynb`
  - renamed reviewer-response notebook
  - keeps the original Figure 2 and Figure 3 code intact
  - appends the additional simulation block for the choice of `h`, the choice of `G`, and the comparison with the GMM-type loss
- `Reviewer_Response_hG_and_Loss_Result.png`
  - uploaded reviewer-response plot used for the manuscript write-up

## Running order

1. Run `Fig02_Fig03_Simulation_Estimation_Error_and_ATE_MSE.ipynb` to reproduce the Figure 2 and Figure 3 simulation results.
2. Run `Fig02_Fig03_Reviewer_Response_hG_and_Loss.ipynb` to reproduce the additional reviewer-response simulation.

## Notes

- The reviewer-response notebook reuses the same DGP and the same `J` grid as the Figure 3 simulation.
- The reviewer-response block compares:
  - MMSCM diagonal loss, `G=10`, uniform weights
  - MMSCM diagonal loss, `G=100`, uniform weights
  - MMSCM diagonal loss, `G=100`, CHF weights with `h=1`
  - MMSCM GMM-type loss, `G=10`, efficient weighting matrix
  - Abadie
- `scpi_pkg` is required only for the SCPI comparison that is already included in the Figure 2 and Figure 3 notebook.

