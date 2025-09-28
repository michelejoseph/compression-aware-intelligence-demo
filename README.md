
# cai mini proof (quick + dirty)

what it shows:
- more compression -> higher recon loss -> lower clf accuracy.
- isr (1 - nmse) tracks accuracy. on this run r ≈ 0.988.

how to run:
- `python cai_proof_min.py` from this folder. it prints lines and saves png/csv under `cai_proof_self/`.

files:
- results.csv
- accuracy_vs_components.png
- accuracy_vs_isr.png

notes:
- data is synthetic (50 feats, 30 informative). pca used as compressor. swap in whatever and check if isr still tracks.
- baseline row (no compression) sits at isr=1.0 by design.
