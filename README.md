# OASIS Emotion Prediction

**Avaneesh Babu, Akira Lonske**

## Hypothesis

The percent composition of dominant colors in an image, combined with its semantic category, can be used to predict the valence and arousal scores assigned to that image by human raters.

## Background

The [OASIS dataset](https://www.benedekkurdi.com/#oasis) (Open Affective Standardized Image Set) is a collection of 900+ color images rated by roughly 800 participants for:

- **Valence** — the degree of positive or negative affective response (1–7 scale)
- **Arousal** — the intensity of the emotional response itself (1–7 scale)

Images are split across four categories: **Animals**, **Objects**, **Scenery**, and **People**.

While prior work has analyzed which image categories tend to elicit particular emotional responses, little has examined the predictive power of *color composition* and *semantic category* alone. This project asks:

> Can valence and arousal be reliably predicted from (1) the percentage of an image occupied by each of 11 perceptually distinct color bins, (2) a binary dominance mask over those bins, and (3) the image's OASIS semantic category?

## Approach

1. **Color classifier** — A logistic regression trained on the [XKCD color naming dataset](https://blog.xkcd.com/2010/05/03/color-survey-results/) maps each pixel's RGB value (in CIELAB space) to one of 11 perceptually distinct color bins: red, orange, yellow, green, blue, purple, pink, brown, gray, teal, tan. Bins with poor separability (black, white, olive, maroon, lavender) were dropped after evaluating per-class F1 scores.
2. **Color features** — Each OASIS image is analyzed pixel-by-pixel using the trained classifier. The result is a percentage-composition vector (fraction of pixels per bin) plus a binary dominance mask.
3. **Semantic features (Experiment 1)** — Use the existing OASIS category labels (Animals, Objects, Scenery, People) as a one-hot semantic feature.
4. **Semantic features (Experiment 2)** — Replace hand-labeled categories with predictions from a pretrained image classification model and measure whether this changes performance.
5. **Models** — A Ridge baseline (linear) and a small MLP (nonlinear, dual-head for valence + arousal) are both trained on the combined feature vectors and compared with 5-fold cross validation. Ridge tests whether the hypothesized signal is recoverable from color + category at all; the MLP tests whether nonlinear feature interactions add anything beyond a linear baseline. Performance is reported as mean squared error per fold.
6. **GUI** — A final interface that accepts any OASIS image and outputs a predicted valence/arousal score alongside a visualization of its color composition.

## Project Structure

```
oasis-emotion-prediction/
├── notebooks/
│   └── final_demo.ipynb       # Polished end-to-end demo (Colab-ready)
├── src/
│   ├── data_loader.py         # Load OASIS images and ratings CSV  [complete]
│   ├── color_classifier.py    # Train + save XKCD color bin classifier  [complete]
│   ├── color_features.py      # Per-image bin composition + dominance mask  [complete]
│   ├── semantic_features.py   # OASIS category encoding and classifier-based semantics
│   ├── model.py               # Ridge baseline + MLP regressor
│   └── train.py               # K-fold training and evaluation pipeline
├── models/
│   └── saved_models/          # Serialized trained models (not tracked in git)
├── data/
│   ├── oasis/                 # OASIS images and ratings CSV (not tracked in git)
│   └── xkcd/                  # XKCD color naming CSV (not tracked in git)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

**OASIS** — Download from [benedekkurdi.com/#oasis](https://www.benedekkurdi.com/#oasis) or [https://db.tt/yYTZYCga](https://db.tt/yYTZYCga) and place images and the ratings CSV inside `data/oasis/`.

**XKCD color naming data** — Required to train the color bin classifier. Download the survey data from the [XKCD color survey blog post](https://blog.xkcd.com/2010/05/03/color-survey-results/) (the post links to the full RGB-name dump), and produce a CSV with columns `r`, `g`, `b`, `term` named `xkcd_teaching.csv` inside `data/xkcd/`.

Both datasets are gitignored and must be obtained separately.

## Training the color classifier

Run this once before anything else. It trains the XKCD-based color bin classifier and saves it to `models/saved_models/color_classifier.pkl`:

```bash
python src/color_classifier.py
```

## Training the emotion model

`--experiment` selects the semantic-feature source:

- **1** — OASIS `Category` labels.
- **2** — pretrained ResNet-50 (ImageNet) predictions mapped back to OASIS categories.

`--model` selects the regressor (`ridge` or `mlp`).

```bash
python src/train.py --csv data/oasis/OASIS.csv --images data/oasis/Images --experiment 1 --model ridge
python src/train.py --csv data/oasis/OASIS.csv --images data/oasis/Images --experiment 2 --model mlp
```

Models are saved to `models/saved_models/` with an `_exp{1,2}` suffix.
