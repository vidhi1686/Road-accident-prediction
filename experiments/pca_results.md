# PCA vs No-PCA Comparison

Generated: `2026-11-19T16:31:08.752821+00:00`
Dataset: `12316` rows, severe prevalence `15.4352%`
Split: train `7881`, validation `1971`, test `2464` (random_state `42`)

## Validation comparison

| Estimator | Variant | n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe precision | Severe recall | Severe F1 | Macro F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-rf | no PCA | — | — | 0.1700 | 0.7661 | 0.6318 | 0.3144 | 0.4375 | 0.3659 | 0.6112 | 0.6783 | 0.3307 |
| baseline-rf | PCA | 40 | 0.9804 | 0.1800 | 0.7763 | 0.6122 | 0.3123 | 0.3750 | 0.3408 | 0.6030 | 0.6458 | 0.3168 |
| baseline-logreg | no PCA | — | — | 0.1500 | 0.5495 | 0.5642 | 0.1894 | 0.5855 | 0.2862 | 0.4785 | 0.6009 | 0.2285 |
| baseline-logreg | PCA | 30 | 0.9528 | 0.1700 | 0.6606 | 0.5761 | 0.2153 | 0.4539 | 0.2921 | 0.5344 | 0.5988 | 0.2271 |

## Held-out test comparison

| Estimator | Variant | n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe precision | Severe recall | Severe F1 | Macro F1 | ROC-AUC | PR-AUC | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-rf | no PCA | — | — | 0.1700 | 0.7484 | 0.6199 | 0.2895 | 0.4342 | 0.3474 | 0.5958 | 0.6765 | 0.3349 | [[1679, 405], [215, 165]] |
| baseline-rf | PCA | 40 | 0.9804 | 0.1800 | 0.7606 | 0.5970 | 0.2831 | 0.3605 | 0.3171 | 0.5860 | 0.6497 | 0.2991 | [[1737, 347], [243, 137]] |
| baseline-logreg | no PCA | — | — | 0.1500 | 0.5353 | 0.5531 | 0.1826 | 0.5789 | 0.2776 | 0.4675 | 0.5890 | 0.2477 | [[1099, 985], [160, 220]] |
| baseline-logreg | PCA | 30 | 0.9528 | 0.1700 | 0.6429 | 0.5554 | 0.1973 | 0.4289 | 0.2703 | 0.5169 | 0.5830 | 0.2312 | [[1421, 663], [217, 163]] |

## PCA component search (validation only)

### baseline-rf

| n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe F1 | Macro F1 | Predicted severe rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.3243 | 0.1600 | 0.6687 | 0.5406 | 0.2486 | 0.5180 | 0.2867 |
| 5 | 0.5455 | 0.1700 | 0.7544 | 0.6047 | 0.3278 | 0.5888 | 0.2111 |
| 10 | 0.7316 | 0.1500 | 0.7255 | 0.6239 | 0.3490 | 0.5875 | 0.2674 |
| 15 | 0.8421 | 0.1600 | 0.7260 | 0.6175 | 0.3415 | 0.5842 | 0.2618 |
| 20 | 0.8955 | 0.1400 | 0.6342 | 0.6143 | 0.3305 | 0.5394 | 0.3922 |
| 30 | 0.9528 | 0.1700 | 0.7504 | 0.6090 | 0.3333 | 0.5899 | 0.2202 |
| 40 | 0.9804 | 0.1800 | 0.7763 | 0.6122 | 0.3408 | 0.6030 | 0.1852 |
| 60 | 0.9985 | 0.1700 | 0.7428 | 0.6139 | 0.3390 | 0.5896 | 0.2349 |

### baseline-logreg

| n_components | Explained variance | Threshold | Accuracy | Balanced acc. | Severe F1 | Macro F1 | Predicted severe rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.3243 | 0.1600 | 0.5307 | 0.5034 | 0.2336 | 0.4477 | 0.4581 |
| 5 | 0.5455 | 0.1600 | 0.5545 | 0.4986 | 0.2244 | 0.4560 | 0.4201 |
| 10 | 0.7316 | 0.1700 | 0.6743 | 0.5681 | 0.2819 | 0.5356 | 0.2993 |
| 15 | 0.8421 | 0.1600 | 0.6007 | 0.5636 | 0.2826 | 0.5030 | 0.4023 |
| 20 | 0.8955 | 0.1700 | 0.6565 | 0.5697 | 0.2851 | 0.5295 | 0.3262 |
| 30 | 0.9528 | 0.1700 | 0.6606 | 0.5761 | 0.2921 | 0.5344 | 0.3252 |
| 40 | 0.9804 | 0.1700 | 0.6540 | 0.5695 | 0.2851 | 0.5284 | 0.3298 |
| 60 | 0.9985 | 0.1500 | 0.5510 | 0.5651 | 0.2869 | 0.4796 | 0.4754 |

## Conclusion

- **baseline-rf**: PCA reduces held-out performance. Test macro F1 0.5958 → 0.5860 (Δ -0.0098); severe F1 Δ -0.0302. Best PCA size: 40 components (98.04% variance).
- **baseline-logreg**: mixed held-out performance. Test macro F1 0.4675 → 0.5169 (Δ +0.0494); severe F1 Δ -0.0073. Best PCA size: 30 components (95.28% variance).

Production reference check (no-PCA Random Forest vs `model/metrics.json`): reproduced exactly.
