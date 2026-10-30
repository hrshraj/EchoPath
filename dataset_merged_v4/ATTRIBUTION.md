# EchoPath training data (dataset_merged_v4): sources, licences and attribution

Generated 2026-10-09. Per-image provenance (source, URL, version, licence, original file, COCO image id and Flickr licence) is in `provenance.csv`. BibTeX entries for every source are in `sources.bib`.

## Licence summary

- **18 Roboflow Universe datasets, 22,118 images: CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Use, adaptation and redistribution are permitted, including commercially, provided each source is credited, the licence is linked, and changes are indicated.
- **COCO 2017, 1,111 images.** Annotations: CC BY 4.0. Images: each keeps its own Flickr licence (see below and `provenance.csv`). Cite Lin et al. (2014).
- No images were captured by the authors. The 13 evaluation videos recorded in the Academic Block Building are not part of this dataset.

## Changes made by the EchoPath authors (applies to every source)

Images renamed to sequential ids; class names remapped to the 17-class EchoPath scheme (person, chair, door, stairs, slide, whiteboard, notice_board, elevator, column, window, water_dispenser, potted_plant, waste_bin, fire_extinguisher, wet_floor_sign, table, monitor); classes outside this scheme dropped; each source's test split merged into training; training images that near-duplicate a validation image removed (perceptual hash, Hamming distance <= 4); datasets merged into one collection. Source-specific changes are listed in the table.

## Roboflow Universe sources (all CC BY 4.0)

| # | Dataset | Author | URL | Version | Updated | Accessed | Classes used | Train | Valid | Source-specific changes | BibTeX key |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | person | Rasheed Riga | https://universe.roboflow.com/rasheed-riga-ilxzv/person-fjw64 | 4 | 2024-02-19 | Apr 2026 | person | 1445 | 199 | person -> person | `person-fjw64_dataset` |
| 2 | chairs | Object Detect | https://universe.roboflow.com/object-detect-zjtnj/chairs-dnhsd | 5 | 2025-06-23 | Apr 2026 | chair | 700 | 99 | Chair -> chair; no predefined split, split 80/10/10 by the authors | `chairs-dnhsd_dataset` |
| 3 | Door | Nyla | https://universe.roboflow.com/nyla/door-ex6s8 | 1 | 2021-12-10 | Apr 2026 | door | 961 | 275 | door -> door | `door-ex6s8_dataset` |
| 4 | Navigation | Dataset | https://universe.roboflow.com/dataset-kenli/navigation-hlzh2 | 1 | 2025-11-01 | Oct 2026 | stairs, elevator, door | 1072 | 198 | stairs -> stairs; elevator, elevator_open, elevators_close -> elevator; door -> door; Socket, button, doorhandle, exit dropped; only images containing stairs or an elevator used | `navigation-hlzh2_dataset` |
| 5 | White-board Detection | Student | https://universe.roboflow.com/student-cbw0s/white-board-detection | 2-4 (identical splits) | 2024-10-31 | Apr 2026 | whiteboard, slide | 2951 | 310 | White-board -> whiteboard; Slide -> slide; Instructor class dropped | `white-board-detection_dataset` |
| 6 | Notice boards iiee | Library | https://universe.roboflow.com/library-nbtst/notice-boards-iiee-ydmbd | 1-3 (identical splits) | 2025-07-21 | Apr 2026 | notice_board | 48 | 17 | notice board -> notice_board | `notice-boards-iiee-ydmbd_dataset` |
| 7 | elevator_door | Elevator Panel detection | https://universe.roboflow.com/elevator-panel-detection/elevator_door-riuxl | 15 | 2026-01-22 | Oct 2026 | elevator | 279 | 19 | door_closed, door_moving, door_opened -> elevator; random 25% sample | `elevator-door-riuxl_dataset` |
| 8 | Column | SD | https://universe.roboflow.com/sd-ug5qc/column-2i5mg | 2 | 2025-05-23 | Apr 2026 | column | 754 | 85 | column -> column; no predefined split, split 80/10/10 by the authors | `column-2i5mg_dataset` |
| 9 | windows | rkdf | https://universe.roboflow.com/rkdf/windows-qvnbq | 1 | 2024-09-23 | Apr 2026 | window | 450 | 9 | windows -> window | `windows-qvnbq_dataset` |
| 10 | Water dispenser | akaike | https://universe.roboflow.com/akaike/water-dispenser | 1 | 2022-12-14 | Apr 2026 | water_dispenser | 115 | 136 | Water Dispenser -> water_dispenser (synthetic composite images) | `water-dispenser_dataset` |
| 11 | Plant Object Detection | Manans Workspace | https://universe.roboflow.com/manans-workspace/plant-object-detection | 4 | 2025-05-07 | Oct 2026 | potted_plant | 121 | 16 | plant -> potted_plant | `plant-object-detection_dataset` |
| 12 | waste-bin | Umer Khan | https://universe.roboflow.com/umer-khan/waste-bin | 1-2 (identical splits) | 2022-08-20 | Apr 2026 | waste_bin | 54 | 5 | waste-bin -> waste_bin | `waste-bin_dataset` |
| 13 | Dustbin Detection | workspace | https://universe.roboflow.com/workspace-yus7k/dustbin-detection-u76yd | 1 | 2025-05-04 | Oct 2026 | waste_bin | 114 | 28 | dustbin -> waste_bin | `dustbin-detection-u76yd_dataset` |
| 14 | Trash_bin_detection | trialyolo | https://universe.roboflow.com/trialyolo/trash_bin_detection-9glxi | 1 | 2025-05-20 | Oct 2026 | waste_bin | 642 | 42 | garbage container -> waste_bin; random 30% sample | `trash-bin-detection-9glxi_dataset` |
| 15 | Fire Extinguisher | Kieran | https://universe.roboflow.com/kieran-xejdv/fire-extinguisher-ftnny | 2 | 2023-09-18 | Apr 2026 | fire_extinguisher | 6353 | 861 | extinguisher -> fire_extinguisher | `fire-extinguisher-ftnny_dataset` |
| 16 | wet floor | FRC 5881 | https://universe.roboflow.com/frc-5881/wet-floor-nhjwl | 1 | 2024-06-05 | Apr 2026 | wet_floor_sign | 1086 | 44 | sign -> wet_floor_sign | `wet-floor-nhjwl_dataset` |
| 17 | table-person | AIfull | https://universe.roboflow.com/aifull/table-person | 8-9 (identical splits) | 2024-12-06 | Apr 2026 | table | 1421 | 111 | table -> table | `table-person_dataset` |
| 18 | Computer monitor | Object detection | https://universe.roboflow.com/object-detection-bwxix/computer-monitor | 1 | 2023-02-21 | Oct 2026 | monitor | 837 | 261 | Computer-monitor -> monitor | `computer-monitor_dataset` |

Note: the author shown is the Roboflow workspace display name, which is how Roboflow's own "Cite this dataset" entry credits it.

## COCO 2017 (Lin et al., 2014)

| Subset | Train | Valid | What was used |
|---|---|---|---|
| Chair images (from v3) | 600 | 200 | Images originally selected for chairs; all COCO labels for person, chair, potted plant, dining table (-> table) and tv (-> monitor) used |
| Indoor potted-plant images (added in v4) | 304 | 7 | Indoor scenes with a potted plant >= 1.5% of the image; only images whose Flickr licence is CC BY 2.0, no known copyright restrictions, or US Government work; same five label types used |

Flickr licences of the COCO images used (number of images):

- Chair images (from v3): Attribution-NonCommercial-ShareAlike License: 239; Attribution-NonCommercial-NoDerivs License: 191; Attribution License: 119; Attribution-NonCommercial License: 101; Attribution-ShareAlike License: 70; Attribution-NoDerivs License: 48; No known copyright restrictions: 1
- Potted-plant images (v4): Attribution License: 309; No known copyright restrictions: 2

Some of the chair images from v3 carry NonCommercial, NoDerivs or ShareAlike Flickr licences. They are fine for academic research and publication; check them (listed per image in `provenance.csv`) before any commercial use or redistribution of the image files.

## Sources excluded

Removed relative to v3 (original source could not be identified, so it could not be attributed): the earlier stairs, elevator, potted-plant and monitor data, and the backpack/handbag data (classes dropped).

Reviewed and rejected:

- yolov5-mq5nd/stair-elevator: architectural floor plans, not photographs
- lena-f7w17/wet-floor-detection1: toy (miniature hospital playset) scenes, not real-world
- test-workspace-re3cc/wfs-test-oz1f9: mostly catalogue product photos
- sarweshs-workspace/dustbin-nslds: contains stock-photo watermarks (redistribution rights doubtful)
- cps-laboratory/-te1rk: printed pictures of elevators taped to walls
- datarock/lift_door: strong fisheye distortion
- perception-01-3e0d2/stairs-without-ground-od and elevator-0iq4p/dataset-elevator-status: inaccurate bounding boxes

## Suggested text for the paper

> No images were captured by the authors for training. The detector was trained on images from COCO 2017 [COCO] and 18 public Roboflow Universe datasets [refs], all released under the CC BY 4.0 licence. We remapped their class labels to our 17-class scheme, dropped classes outside it, merged each dataset's test split into training, removed training images that were near-duplicates of validation images, and combined the datasets; the full list of sources, versions and changes is given in Supplementary Table S1.

