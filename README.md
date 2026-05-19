# Learning Aesthetics

Comparing vision and vision-language embeddings for predicting human aesthetic ratings of artworks.

This project studies whether pretrained image models can predict how people rate artworks aesthetically. It compares classic vision-only backbones and newer vision-language / multimodal models by extracting one embedding per artwork, training the same Ridge regression pipeline for each model, and analyzing both prediction accuracy and error patterns.

## Motivation

Human aesthetic judgments may depend on both low-level visual features, such as color and texture, and higher-level semantic/contextual cues. This project asks:

- Can pretrained machine-learning models predict human aesthetic ratings for artworks?
- Do vision-language models outperform vision-only models?
- Do different model families make the same kinds of mistakes, or complementary ones?

## Dataset And Privacy Note

The analysis uses a collaborator-provided artwork dataset with 800+ images and human aesthetic ratings.

The full artwork images, thumbnails, previews, rating file, and extracted embeddings are **not included** in this public version of the repository. The artwork dataset may have collaborator, copyright, or data-sharing restrictions, so this repository only includes code, aggregate results, and the project poster.

Local qualitative HTML galleries were generated during analysis, but the image assets are intentionally omitted from this public version.

See [`data/README.md`](data/README.md) for the expected local data layout.

## Models Compared

Vision-only models:

- AlexNet
- VGG16
- ResNet50
- InceptionV3
- ViT-B/16
- DINOv2

Vision-language / multimodal models:

- CLIP
- BLIP-2
- SigLIP2
- ImageBind
- GIT
- Florence-2

## Method

For each model:

1. Load a pretrained image encoder.
2. Extract one fixed-length embedding per artwork.
3. Align embeddings with human aesthetic ratings from a MATLAB `.mat` label file.
4. Train the same regression pipeline:

   ```text
   L2 normalization -> StandardScaler -> RidgeCV
   ```

5. Evaluate all models on the same train/test split.
6. Compare MSE, MAE, and R².
7. Analyze per-image prediction errors and model-pair error correlations locally.

## Results

The final shared test-set metrics are available in:

[`results/summary_metrics_TEST.csv`](results/summary_metrics_TEST.csv)

Top-performing models in the final test split:

| Model | MSE | MAE | R² |
|---|---:|---:|---:|
| ImageBind | 0.0275 | 0.1326 | 0.8702 |
| BLIP-2 | 0.0282 | 0.1291 | 0.8672 |
| SigLIP2 | 0.0292 | 0.1337 | 0.8624 |
| CLIP-B/16 | 0.0323 | 0.1418 | 0.8479 |
| CLIP-B/32 | 0.0371 | 0.1475 | 0.8253 |

Overall, vision-language and multimodal embeddings produced lower prediction error than most vision-only CNN or transformer features.

## Error Pattern Analysis

Beyond model accuracy, the project compares per-image prediction errors across models. Pearson correlations between error vectors show whether two models fail on similar artworks.

The analysis suggests that vision-only and vision-language models can make systematically different mistakes. This is useful because complementary error patterns may point to different visual or semantic cues used by each model family.

The public repository does not include the per-image galleries or preview images because those would expose collaborator-provided artwork images.

## Sanity Check

A random-label sanity check was run by shuffling the human ratings and rerunning the training pipeline. Model R² scores dropped close to zero, supporting that the original models were learning a real relationship between artwork embeddings and human ratings rather than fitting noise or file-order artifacts.

## Poster

The project poster is included here:

[`docs/MosAI_Poster.pdf`](docs/MosAI_Poster.pdf)

## Repository Structure

```text
.
├── README.md
├── data/
│   └── README.md
├── docs/
│   └── MosAI_Poster.pdf
├── results/
│   └── summary_metrics_TEST.csv
├── src/
│   ├── unified_regress.py
│   ├── rank_gallery.py
│   ├── error_matrix_and_dimreduce.py
│   ├── heatmap.py
│   ├── pearson_correlation.py
│   ├── shuffled_label_sanity_check.py
│   └── extract/
└── requirements.txt
```

## Main Scripts

- `src/unified_regress.py`: trains a Ridge regression model on one model's embeddings and reports MSE, MAE, and R².
- `src/rank_gallery.py`: runs all models on the same train/test split and generates ranking/error CSVs plus local HTML galleries.
- `src/error_matrix_and_dimreduce.py`: builds a per-image model-error matrix and projects it to 2D with PCA.
- `src/heatmap.py`: visualizes model-wise error correlations.
- `src/pearson_correlation.py`: groups error correlations by model-family pairing.
- `src/shuffled_label_sanity_check.py`: validates the pipeline using shuffled labels.
- `src/extract/`: model-specific embedding extraction scripts.

## How To Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the artwork image folder, label `.mat` file, and embedding folders as described in [`data/README.md`](data/README.md).

3. Update the path configuration at the top of the relevant script.

4. Run a single-model regression:

   ```bash
   python src/unified_regress.py
   ```

5. Run the shared train/test ranking and gallery workflow locally:

   ```bash
   python src/rank_gallery.py
   ```

## Interview Talking Points

- Built a reproducible comparison pipeline for pretrained visual and multimodal embeddings.
- Used a shared train/test split so model comparisons are fair.
- Combined representation learning with classical regression to predict human aesthetic ratings.
- Compared both accuracy metrics and model error patterns, not just leaderboard scores.
- Added a shuffled-label sanity check to test for leakage or spurious fitting.
- Built local HTML galleries to inspect which artworks each model rated well or poorly while keeping collaborator images out of the public repository.

## Limitations

- The full artwork dataset, human ratings, per-image previews, and embeddings are not included in this public repository.
- Several scripts still use local absolute paths and should be updated before running in a new environment.
- The project is an experimental research workflow rather than a packaged Python library.
