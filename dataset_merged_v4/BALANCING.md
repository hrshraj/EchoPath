# Class balancing (training split only; validation untouched)

1. Near-duplicate leakage: training images within dHash Hamming distance 4 of any validation image were removed (2727 images, listed in REMOVED_leakage.csv).
2. Undersampling of over-represented, lower-importance classes (images containing only that class): fire_extinguisher: 4588 images dropped; monitor: 234 images dropped; whiteboard: 1463 images dropped. Targets: fire_extinguisher ~3000 labels, monitor ~1000 labels, whiteboard/slide-only images 50%.
3. Importance-weighted repeat-factor sampling: tier-1 target 3000 labels, tier-2 target 1500; r_c = min(3, sqrt(target / count)); image repeat = max r_c over its labels, stochastic rounding (seed fixed). Tier 3 and water_dispenser (synthetic) are never repeated. Implemented as train_balanced.txt (paths repeated); no image is duplicated on disk.

Tiers: 1 = fall/collision hazards (stairs, wet_floor_sign, waste_bin, column, chair, table, person); 2 = landmarks/minor obstacles (door, elevator, potted_plant, water_dispenser, fire_extinguisher); 3 = informational (whiteboard, slide, notice_board, monitor, window).

| Class | Tier | Train labels | After undersampling | Repeat factor | Effective labels (balanced list) | Valid labels |
|---|---|---|---|---|---|---|
| person | 1 | 3676 | 3676 | 1.00 | 3922 | 743 |
| chair | 1 | 2807 | 2807 | 1.03 | 3163 | 666 |
| door | 2 | 1016 | 1016 | 1.22 | 1241 | 292 |
| stairs | 1 | 1102 | 1102 | 1.65 | 1817 | 138 |
| slide | 3 | 1472 | 736 | 1.00 | 736 | 147 |
| whiteboard | 3 | 2846 | 1431 | 1.00 | 1431 | 301 |
| notice_board | 3 | 48 | 48 | 1.00 | 48 | 17 |
| elevator | 2 | 524 | 524 | 1.69 | 882 | 95 |
| column | 1 | 2458 | 2458 | 1.10 | 2725 | 252 |
| window | 3 | 529 | 529 | 1.00 | 529 | 10 |
| water_dispenser | 2 | 115 | 115 | 1.00 | 115 | 136 |
| potted_plant | 2 | 852 | 852 | 1.33 | 1119 | 75 |
| waste_bin | 1 | 862 | 862 | 1.87 | 1604 | 78 |
| fire_extinguisher | 2 | 10624 | 2954 | 1.00 | 2954 | 2175 |
| wet_floor_sign | 1 | 1057 | 1057 | 1.68 | 1779 | 43 |
| table | 1 | 2204 | 2204 | 1.17 | 2643 | 259 |
| monitor | 3 | 1347 | 1018 | 1.00 | 1044 | 436 |

Training images on disk: 20307; unique images in balanced list: 14022; balanced list length: 17110.
