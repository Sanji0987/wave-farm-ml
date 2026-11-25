# References (25 Nov 2026)

*For the deck, the demo notebook and the EC4 report. Status: **checked** = title, authors and venue read from the paper's own title page (PDF in the project or the uploads); **standard** = well-known citation, not opened; **check** = not yet verified, confirm before the report.*

## Data
| Reference | Used for | Status |
| --- | --- | --- |
| Neshat, M., Alexander, B., Sergiienko, N., & Wagner, M. (2020). *Large-scale Wave Energy Farm* [Dataset]. UCI Machine Learning Repository. doi:10.24432/C5GG7Q | the layout dataset (Perth / Sydney × 49 / 100 buoys) | check (authors, year and DOI from the demo chat; confirm on the UCI page) |
| Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. | ERA5 waves (`swh`, `mwp`, `pp1d`, `mwd`), 2016–17 | standard |
| GEBCO Compilation Group (2026). *GEBCO gridded bathymetry* (downloaded 6 Sep 2026). | depths, the 30–100 m band, land mask | check (confirm the official grid name, e.g. "GEBCO_2026 Grid", on gebco.net) |

## Wave farms, layouts and models
| Reference | Used for | Status |
| --- | --- | --- |
| Neshat, M., Abbasnejad, E., Shi, Q., et al. (2019). Adaptive neuro-surrogate-based optimisation method for wave energy converters placement optimisation. arXiv:1907.03076. | prior surrogate work on WEC layouts; device and site set-up | checked title; check author order on arXiv (the demo notebook's author list for this paper is wrong) |
| Neshat, M., Alexander, B., Sergiienko, N. Y., & Wagner, M. (2020). Optimisation of large wave farms using a multi-strategy evolutionary framework. arXiv:2003.09594 (GECCO 2020). | the optimisers that produced the dataset; 49 / 100-buoy farms | checked (title, first authors) |
| Neshat, M., et al. (2024). *Energy* 304, 132122. doi:10.1016/j.energy.2024.132122 (demo notebook gives the title "MetaWave Learner"). | cross-site transfer done before us (§4.4, Table 8 per October notes) | check (title and authors) |
| Amini, E., Golbaz, D., Amini, F., et al. (2020). Performance investigation on the different wave energy converters layouts in real wave scenarios. arXiv:2007.14012. | device parameters (PTO stiffness / damping), layouts in real seas | checked (title, first authors) |
| Ganapathi Subramanian, P. (2026). Surrogate modelling of large-scale wave energy converter arrays using geometric feature engineering. *Journal of Ocean Engineering and Marine Energy*. doi:10.1007/s40722-026-00522-0. | "the published pipeline": R² 0.972 on Perth 49 with a random split | checked |
| Sergiienko, N. Y. (2019). *WEC array simulator* (v1.0.2). MATLAB Central File Exchange #71840. | the simulator (run in GNU Octave) | checked (copyright line in the code's licence) |

## Wave energy resource and devices
| Reference | Used for | Status |
| --- | --- | --- |
| Babarit, A. (2015). A database of capture width ratio of wave energy converters. *Renewable Energy*, 80, 610–628. | typical capture-width ratio, 16 ± 10% for heaving devices | checked (title, venue); the 16 ± 10% figure itself not re-read |
| Zheng, C., et al. (2021). Global oceanic wave energy resource dataset — with the Maritime Silk Road as a case study. *Renewable Energy*, 169, 843–854. | regional wave-resource context (Arabian Sea / Indian Ocean) | checked (title, venue) |
| Ocean Energy Europe (2024). *Ocean Energy: Key trends and statistics 2024.* | Europe since 2010: 13.5 MW installed, 12.6 MW decommissioned, 0.83 MW in the water | check (edition and figures) |

## Software
| Reference | Used for | Status |
| --- | --- | --- |
| Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. | HistGradientBoosting, GroupKFold, logistic regression, random forest, ridge | standard |
| GNU Octave (version 8.4). | running the simulator | standard |

## Not to cite without checking
- Civitavecchia "planned 2.5 MW, ~20 kW installed" (from OES 2017 per October notes).
- Shahriar & Habib 2024, Afolabi 2025, Bessonova 2025, Anusree 2024: mentioned in October notes, never verified.
