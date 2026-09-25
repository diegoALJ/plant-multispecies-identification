# Project Notes

## Current experiment status

- First official DINOv2 model was run with 5x5 tiling plus full image.
- Saved score artifacts were used for post-processing.
- Conservative top-k submissions performed better than longer prediction lists.
- Best early public score observed: around 0.29 with top5.

## Next planned experiment

Run the second official PlantCLEF DINOv2 checkpoint and save the same artifacts in a separate folder, then ensemble both score matrices in the post-processing notebook.
