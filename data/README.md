# Data Notes

The full artwork dataset, human ratings, extracted embeddings, thumbnails, and preview images are not included in this public repository.

This is intentional: the artwork images come from a collaborator-provided dataset and may not be publicly redistributable.

Expected local inputs:

```text
artwork_images/
  image_001.jpg
  image_002.jpg
  ...

image_mean_rating_shuffled2.mat

MODEL_NAME-embeddings/
  embeddings.npy
  filenames.txt
```

The `.mat` label file should contain:

- `image_names`: artwork filenames or filename stems
- `mean_score` or `mean_rating`: human aesthetic rating for each artwork

Each embedding folder should contain:

- `embeddings.npy`: an `N x D` array of image embeddings
- `filenames.txt`: one filename per row, aligned to `embeddings.npy`

The scripts align labels and embeddings by filename stem, then remove missing or non-finite labels before training.
