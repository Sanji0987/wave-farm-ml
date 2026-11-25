# Results: wave energy project (v7, authoritative)

*16 Nov 2026. Supersedes `final_results_2026-10-15.md` (v6), `revised_results_2026-10-15.md` and `project_audit_2026-10-09.md`. Tags: **[RC]** = recomputed from the raw data in the 13 Nov re-check and confirmed; **[RC-fixed]** = recomputed and corrected; **[not RC]** = not yet re-checked (value as reported in October / 5 Nov). Open items: `recheck_status.md`.*

## 1. Dataset (UCI Large-scale Wave Energy Farm, Neshat et al.)
- Four files, Perth or Sydney waves × 49 or 100 buoys: **63,602 rows, 19,005 unique layouts**. Duplicate share per file: Perth 49 71.7%, Sydney 49 73.2%, Perth 100 62.4%, Sydney 100 46.8%. [RC]
- Perth 49: **36,043 rows, 10,218 unique layouts**, 71.7% exact repeats. No layout appears again with buoys in a different order. [RC]
- Farm area: 1,000 × 1,010 m for the 49-buoy files; **1,450 × 1,450 m for the 100-buoy files**. [RC-fixed]
- Spacing: 0.5% of unique Perth 49 layouts (56) have a pair closer than 50 m; 52 have buoys 6.08 m apart. 61% of Sydney 49 layouts break the 50 m rule. [RC]
- Best Perth 49 layout: Total_Power **4,177,659 W** (85,258 W per buoy, q = 0.885). Its own per-buoy columns sum to 4,181,731 W, 4.1 kW more. Worst: 3,388.9 kW. [RC]
- Lone-buoy power implied by the qW column: **Perth 96,374 W, Sydney 105,788 W** (each within 1 W). Retire October's 96,388 / 105,781 W and 96.39 / 105.78 kW. [RC]
- Perth 49 has 235 groups of identical coordinates (1,777 rows) carrying different powers (up to 7.7 kW per buoy); none in the top 5%. Total_Power differs from the sum of per-buoy powers by up to 4.1 kW (Perth 49). Perth 100: the best row (0-based 850) carries exactly the power vector of another layout (row 2977). [RC]

## 2. Near-twins and leakage
- **Near-twin** = a layout identical to another except for one moved buoy (48 of 49 positions the same).
- **156,649 twin pairs**, median move **564 m**. 154,846 of them (98.8%) sit inside one family; the other 1,803 pairs have a median move of 100 m. [RC-fixed: v6 said 2,252 pairs and 71 m because of a bug in the pair code]
- **8,051 families** (7,596 single layouts); largest **557 layouts (5.5%)**: all share the same 48 positions, contain the best layout, and span 4,125–4,178 kW. 25.7% of layouts have at least one twin. [RC]
- Leak share: with exact repeats grouped in 5 random folds, **about 23.5%** of test layouts have a twin in training (23.2–23.7% over 20 seeds). [RC]
- Example pair (deck slide 5): one buoy moved 75 m, 48 positions shared, 4,135.9 vs 4,137.2 kW. [RC] It is a small move by the true median.

## 3. Fair testing (the headline) [not RC]
| Perth 49, farm power R² | Random rows | Exact repeats grouped | Twin clusters grouped |
| --- | --- | --- | --- |
| Published pipeline (Subramanian; our re-implementation; paper 0.972) | .970 | .954 | .820 (0.82–0.84 across seeds) |
| 52 shape features | .984 | .977 | .937 |
| Raw coordinates | .966 | .954 | .793 |
| Per-buoy model (5 Nov) [RC] | not run | .982 | .961 (with the 48-of-49 families) |

- Sydney 49: published .997 / .990 / .970; shape .996 / .989 / .977; raw .989 / .979 / .950.
- About two-thirds of the drop is the elite family; held out, that fold scores R² 0.53, the others 0.89–0.92. Elite under-prediction: shape features 1.5–1.7 kW per buoy, raw coordinates 3.9–4.7, published pipeline 3.7, per-buoy model 1.6 [RC].
- **Caveat that must stay with the headline:** the .820 / .937 / .793 column was measured in October with "twin-10 m" clusters (buoys within 10 m per device on average after optimal matching), run at 120 boosting rounds / learning rate 0.1. The deck illustrates and counts twins with the 48-of-49 definition. The headline has not been re-tested under that definition.

## 4. Per-buoy model (demo and loop)
- HistGradientBoosting, 300 trees (250 in CV), learning rate 0.08, ≤31 leaves, ≥50 samples per leaf; 40 features per buoy in the wave frame; target = buoy power ÷ 96,374 W.
- 5-fold CV, exact layouts grouped: farm R² **0.982**, ρ 0.991, MAE 12.5 kW, elite bias −0.24 kW/buoy. Families grouped: **0.961**, ρ 0.989, MAE **16.2 kW** (≈0.4% of farm power), elite bias **−1.61**. Final model, 15% of families held out: R² 0.981, MAE 11.9 kW (1,437 layouts). [RC, identical re-run]
- Demo presets: best layout model 4,153.9 vs dataset 4,177.7 kW; Unseen A 4,123.1 vs 4,104.9. Hill climb from the best layout claims ≈4,253 kW. [not RC]

## 4b. Linear vs quadratic vs trees (17 Nov, same 5-fold split with whole twin families held out) [RC]
| Model (inputs) | Farm R² test | Train | Mean error | Error on true top 5% | True top 5% found | Elite bias per buoy |
| --- | --- | --- | --- | --- | --- | --- |
| Linear ridge (40 per-buoy features) | 0.846 | 0.858 | – | – | – | – |
| Quadratic ridge (all pairs of the 40) | 0.932 | 0.949 | 23.5 kW | 79.1 kW | 20.5% | −1.62 kW |
| Boosted trees (40 features; ours) | 0.961 | – | 16.2 kW | 78.6 kW | 19.0% | −1.60 kW |
| Linear ridge (98 raw coordinates, farm level) | 0.671 | 0.771 | – | – | – | – |
| Boosted trees, default settings (98 raw coordinates) | 0.711 | 0.987 | – | – | – | – |

- Trees beat quadratic overall: mean error 7.3 kW lower (95% CI 6.4–8.0 kW, bootstrap over families); trees closer on 64% of layouts; fold R² 0.921–0.979 vs 0.907–0.942.
- On the best layouts every model is ~79 kW off (the hidden elite family); model choice isn't the bottleneck, data coverage is. Hence the simulator loop.
- On raw coordinates trees overfit (train 0.987, test 0.711). Note: 0.711 is under the 48-of-49 families with default settings; October's 0.793 used twin-10 m clusters. Not a like-for-like re-test.

## 5. Transfer (per-buoy model) [not RC]
| Trained on → tested on | ρ | True top 5% found | Power level R² |
| --- | --- | --- | --- |
| Perth 49 → Perth 100 | 0.97 | 66% | 0.94 debiased (raw 0.56–0.84) |
| Sydney 49 → Sydney 100 | 0.95 | 69% | biased |
| Perth → Sydney | 0.74–0.75 (0.47–0.81 across methods) | 24–27% (chance 5%) | negative (best −0.96) |
| Sydney → Perth | −0.12 | – | – |

## 6. Geometry
- Front / middle / back thirds (ranked up-wave): **92.7 / 80.8 / 64.3 kW** per buoy; back −30.6%. [RC]
- Buoy with no other buoy within ±30° of up-wave: **98.5 kW**, others **72.5 kW**. [RC]
- Wave direction in the layout frame: the team uses **55.3°**; the data's own best fit (power vs up-wave position) is 53.1°, with 51–55° equally good. [RC]
- **Orientation of the best Perth layouts: rows run along about 145–147°, i.e. across the waves; the direction perpendicular to the rows is about 55–57°.** Best layouts = a wide wall across the swell. [RC-fixed: v6 said "the layout grid (≈56°) lines up with the wave direction"]
- Up-wave clearance worth ≈ +1.6 kW per 50 m (model-based). [not RC]
- Sydney 49: mean pairwise buoy distance alone gives R² 0.92 (linear 0.916, 5-fold CV). [RC]

## 7. Static models can't design [not RC]
- Layout-level tree model searched for its best layout: 84,889 W per buoy vs 85,258 in the data (HGB's structural bound 91,944, so the ceiling is empirical).
- GP vs HGB: 0.9587 vs 0.9576 (exact grouped), 0.9312 vs 0.9277 (twin grouped). GP ±2σ covers 93% of layouts, 1% of elite layouts.

## 8. Simulator (WEC Array Simulator, File Exchange #71840, GNU Octave 8.4)
- Fitted PTO, constant per tether: k ≈ 407,400 N/m, d ≈ 96,500 N·s/m; interaction cut-off kR ≈ 200 (default 100). Bundled PTO 2–3.5× too strong. [not RC]
- Lone buoy with the original code and fitted PTO: **Perth 96,372 W, Sydney 105,814 W** (dataset 96,374 / 105,788). [RC]
- Sea-state tables keep 79.5% (Perth) and 79.6% (Sydney) of records. Perth table: directions 217.5–262.5°, Hs 2–5 m, Tp 10.5–16.5 s; Sydney: 22.5–202.5°, Hs 1–3 m, Tp 5.5–12.5 s. [RC]
- Best Perth 49: 4,177,659 vs 4,176,339 W (−0.03%; 4,176,607 at kR 200). Best Sydney 49: 4,145,252 vs 4,152,603 W (+0.18%; −0.06% at kR 200). 23 held-out layouts within ±0.2%. 55–171 s per 49-buoy layout (vectorised). [not RC]
- A consistency check, not validation: PTO fitted on the two best layouts; climate not held out.

## 9. Resonance and Perth vs Sydney [not RC]
- Resonance ≈ 7.6 s. Capture width 9.7 m at Tp 8 s, 3.3 m at 12 s, 1.0 m at 16 s.
- Perth 50.8 kW/m, CW 1.9 m; Sydney 15.3 kW/m (19.4 full hindcast), CW 6.9 m. Perth 2.6–3.3× the wave power, one buoy absorbs ~9% less.
- Linear, unconstrained model: capture-width ratio 60–75% at the Indian Ocean sites vs 16 ± 10% typical (Babarit 2015). All energy is an upper bound.

## 10. Sites (ERA5 2016–17, 30–100 m deep)
- ERA5 file: 0.25° grid, 101 × 101, 8,819 cells with data (all complete). Coastal cells: 41,396 on a 0.05° grid; **5,663 in the band, 5,422 with wave data**, 4,355 unflagged; band wave power 1.1–17.9 kW/m. [RC]
- Wave power = 0.49·Hs²·Te with Te = `mwp`. Old 20.8 kW/m = 17.87 ÷ 0.857 (the EC1 error). [RC]
- Six sites (wave power; Te = time-mean of `mwp`; depth; distance offshore), all in the band and unflagged [RC]:
  1. 7.61N 77.96E, Gulf of Mannar / Laccadive Sea, SE of Kanyakumari: 13.0 kW/m, 7.97 s, 78 m, 67 km. Ranks 46th of the band by wave power.
  2. 8.01N 78.26E, Gulf of Mannar off Tamil Nadu: 10.93, 7.63 s, 98 m, 46 km.
  3. 7.56N 77.16E, Laccadive Sea SW of Kanyakumari: 12.26, 8.05 s, 83 m, 66 km.
  4. 6.61N 81.96E, SE Sri Lanka: 12.32, 8.39 s, 33 m, 21 km.
  5. 6.01N 81.41E, S Sri Lanka: 12.98, 8.58 s, 88 m, 24 km.
  6. 6.51N 93.81E, Nicobar Islands: **17.87 kW/m, highest in the band**, 9.87 s, 68 m, 28 km.
- In this data ERA5's own peak period `pp1d` ≈ 1.23–1.24 × `mwp` (median over the band), longer than the Tp = `mwp`/0.857 = 1.17 × `mwp` the energy method assumes. That is why using `pp1d` lowers site 1's energy. [RC]
- Energy for 49 buoys (absorbed): site 1 **30.4 / 31.6 / 32.7 GWh/yr** at q 0.80 / 0.83 / 0.86, CW 6.81 m; ≈21 (20.0–21.5) and rank 16 with `pp1d`; honest band ≈20–33; ≈23 delivered (×0.8×0.9, illustrative). Site 2 28.3–30.4; site 3 27.5–29.5; sites 4–5 ≈27–29; Nicobar 23.8–25.4, rank 15. q 0.80–0.86 in the sites' own waves (0.64 if badly oriented). [not RC]
- **Independent sites: roughly 150–220 at least 40 km apart** depending on selection order and flag filter (122–166 at 50 km). [RC-fixed: v6's 114–167 at 40 km doesn't reproduce]

## 11. Closed loop and controls (42 simulations each, kR 100) [not RC; cannot be re-run, layouts lost]
- Seed 1 best-so-far: 4151.3, 4155.1, 4176.0, 4194.1, 4194.1, 4201.0, 4201.0 kW vs dataset best re-simulated 4,176.3 kW. At kR 200: 4,201,172 vs 4,176,607 W (+24.6 kW, +0.59%); 10 of 42 proposals beat it; new layout 39.7 m from any dataset layout.
- Model MAE by round, retrained: 122.4, 49.2, 33.3, 60.6, 27.9, 21.7, 29.3 kW; never retrained: 122.4, 71.2, 55.5, 91.4, 53.9, 40.2, 49.7 (1.45–1.93× worse from round 2).
- Seed 2: +6.3 kW (+0.15%), 5/42. Seed 3: +22.1 kW (+0.53%), 11/42. Random s1 −0.6, s2 +0.6 (1/42); geometry rules −3.0 kW (0/42). At kR 200 loop gains +24.6 / +5.5 / +21.5 (mean 17.7, SD 9.9). Worst loop beats best control by 5.7 kW; 3 vs 3, p = 0.05 one-sided.
- Model ≈ +47 to +54 kW optimistic about its own proposals. Loop proposals moved a median of 13–16 buoys, random 1–3.
- Headline: "Across three runs the loop beat the dataset's best layout by +0.15% to +0.59%; model-free search with the same budget managed −0.07% to +0.01%."

## 12. Model 1 (ERA5 trust screen; kept as a screen, 24 Nov)
- 94 stations, 23 positive (|relative mean Hs bias| ≥ 8%). Distance to coast alone, unfitted: **AUC 0.79**. [RC]
- RF − distance, paired: −0.02 (95% CI −0.09 to +0.04): no significant difference. Threshold chosen on the test AUC gave the old 0.80; fixed in advance it is 0.77 (RF) / 0.78 (LR). [not RC]
- Use: a screen on the 41,396 coastal cells; all six recommended sites are unflagged. [RC]
- Honest framing: "ERA5's wave-height bias concentrates near shore and in shallow water; the model learns that, about as well as the distance rule."
- The `flagged` column in `msr_cells.csv` marks 13.4% of cells; re-fitting the model gives 1–9% depending on method. Fitted models (28 features): AUC 0.77 (RF) / 0.78 (LR) [not RC; with the 5 features in `station_summary.csv` the re-check gets 0.68 / 0.71].

## 13. Corrections since EC1 (for the deck's honesty slide)
q-transfer dropped · wave power 20.8 → 17.9 kW/m · energy 12–41 → 20–33 GWh/yr · ERA5 wave grid ~40 km not ~28 km · q measured per site · Model 1 dropped · Civitavecchia planned 2.5 MW, ~20 kW installed · OEE 2024: 13.5 MW installed since 2010, 12.6 MW removed, 0.83 MW in the water [not RC] · Neshat et al. 2024 already tested cross-site transfer [not RC].
