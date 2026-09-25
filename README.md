# PlantCLEF2026 - Multi-Species Plant Identification

This repository contains my working pipeline for the [PlantCLEF2026 @ LifeCLEF & CVPR-FGVC Kaggle competition](https://www.kaggle.com/competitions/plantclef-2026).

The goal is to identify all plant species visible in high-resolution vegetation plot images. In practice, this is a hard multi-label computer vision problem: the model has to look at a full quadrat image and return a list of species IDs.

## Project overview

This project started as a Kaggle experiment focused on biological image analysis. The main idea was to use the official DINOv2-based PlantCLEF models, run tiled inference over the large test images, save reusable score matrices, and then iterate quickly with post-processing and ensemble strategies.

The repository is not meant to be a full MLOps system. It is a clean ML Engineering-style version of the Kaggle workflow, organized so the main pieces can be reused and improved.

## Dataset

The competition uses:

- Single-plant training images with species labels.
- High-resolution test images of vegetation quadrats.
- A list of official `species_id` values.
- Metadata for the test quadrats.

The main challenge is the domain shift: the training data shows individual plants, while the test images contain mixed vegetation, soil, rocks, shadows, frames, and other objects.

## Important data availability note

The image data is not included in this repository.

To reproduce the experiment, you need to join the Kaggle competition, accept the competition rules, and add the required datasets and official models through Kaggle Inputs. The training and test images are large, so this repo only keeps code, configuration files, and lightweight project structure.

## Modeling approach

The current approach is:

1. Load the official PlantCLEF DINOv2 checkpoint.
2. Split each test image into a fixed grid of tiles plus the full image.
3. Run tiled inference on the model.
4. Aggregate tile scores into image-level species scores.
5. Save score matrices for later use.
6. Generate multiple submissions using different top-k, threshold, blending, and ensemble strategies.

The first strong baseline used a 5x5 grid plus full-image inference. Post-processing showed that shorter prediction lists performed better, suggesting false positives were a major issue.

## Key results

Current public submission experiments from the first model:

- `top12`: around 0.22
- `top8`: around 0.25
- `top5`: around 0.29

These results are early baselines. The next planned step is to run the second official model and ensemble both sets of saved scores.

## Repository structure

```text
.
├── configs/                  # YAML configuration files
├── src/                      # Reusable Python modules
├── notebooks/                # Kaggle notebooks are added manually here
├── submissions/              # Optional local submission files
├── outputs/                  # Local generated artifacts, ignored by git
├── assets/                   # Optional images for README or reports
├── requirements.txt
├── .gitignore
└── README.md
```

## Main notebooks

The actual Kaggle work was organized around three notebooks:

1. `01_EDA.ipynb` - metadata and image inspection.
2. `02_FE_Modeling_DINOv2_Tiling_Inference.ipynb` - tiled inference and score export.
3. `03_Postprocessing_Submissions_From_Saved_Scores.ipynb` - score blending, thresholds, dynamic top-k, and future ensemble.

The notebooks are not included in this package. They can be added later inside the `notebooks/` folder.

## Organizers

This competition is part of LifeCLEF 2026 and the FGVC13 workshop at CVPR 2026. The challenge is hosted on Kaggle as PlantCLEF2026.

## Disclaimer

This is a personal learning and portfolio project. It is not an official baseline from the competition organizers. The code is shared for reproducibility and as a record of the experimentation process.
