# Round 8 — Paper-ready statistics summary

- configs: 12
- seeds: [42, 123, 2024]

> **PLACEHOLDER — CPU dry-run**
>
> `stats.json` has `placeholder_stats=true`. All numbers in this document
> come from the CPU placeholder SR proxy in
> `scripts/training/train_dummy_batch.py` (a linear function of the
> dummy-batch training loss), **not** from a LIBERO benchmark success
> rate.
>
> For real paper numbers: run `scripts/training/run_ablation.sh` on an
> RTX 5070 with GPU eval, then rerun `scripts/paper/aggregate_ablation.py`
> and this script.


## Main results (LIBERO-Long)

- Baseline (PhaseQFlow++): 56.9 ± 1.7% (3 seeds)
- Full system (Ident + A + B + C + PCAR): 61.9 ± 4.4%
- Absolute improvement (full vs baseline): **+5.0 pp (p=1.000)**

- PACE only (Ident + A + B + C, no PCAR): 59.5 ± 9.2% (+2.6 pp, p=0.962)
- PCAR only (Ident + PCAR, no PACE): 55.8 ± 5.8% (-1.1 pp, p=0.815)

## Ablation insights

### Removing one component at a time (starting from full)

- Remove PCAR (full → pace): +2.4 pp
- Remove PACE-A (full → bc + PCAR proxy): +4.2 pp (bc+PCAR wasn't run; bc used as an approximation)
- Remove PACE-B (full → ac + PCAR proxy): +4.2 pp (ac+PCAR wasn't run; ac used as an approximation)
- Remove PACE-C (full → ab + PCAR proxy): +5.2 pp (ab+PCAR wasn't run; ab used as an approximation)
- Remove Identifiability (full → pcar_noident): +5.0 pp

### Cross-component interaction (vs sum of singles)

- A + B vs A / B (marginal over ident): synergy = -1.8 pp
- A + C vs A / C: synergy = -0.1 pp
- B + C vs B / C: synergy = +1.8 pp

> Positive synergy means the combined effect exceeds the sum of
> individual effects; negative means the components conflict or have
> saturated.

## Spatial control (LIBERO-Spatial)

LIBERO-Spatial tasks typically have ≤ 2 phases per task, so the
Phase-Centric innovations (especially the PACE-C curriculum) **should
not** visibly lift SR — this is the counter-check for the phase-centric
claim. (If Spatial also lifts a lot, it means we're measuring a "all
data augmentation helps" common cause, not phase structure itself.)

- PACE-C only (spatial): Δ = **-1.9 pp (p=1.000)**
- PACE full (spatial): Δ = -0.5 pp (p=0.196) (long: +2.6 pp, p=0.962)

> If p > 0.05 and Δ_spatial << Δ_long, the hypothesis that "the
> phase-centric gain comes from phase structure" holds.

## Robustness: PCAR without Ident

- PCAR + Identifiability (pcar_only): 55.8 ± 5.8%
- PCAR without Identifiability (pcar_noident): 56.9 ± 2.5%

If pcar_noident << pcar_only, PCAR's replacement power depends on the
identifiable phase latent as its signal source. Otherwise the PCAR β_t
signal is robust on its own.

## Raw CSV / figure links

- `artifacts/ablation/ablation_table_long.csv` — seeds × configs SR matrix
- `artifacts/ablation/ablation_table_spatial.csv` — same for LIBERO-Spatial
- `artifacts/ablation/stats.json` — per-config mean/std/CI + Δ/p
- `paper_figures/fig1_main_bar.png` / `fig2_long_vs_spatial.png` / `fig3_beta_vs_sr.png`
- `paper_figures/ablation_table.tex` — booktabs LaTeX
