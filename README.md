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

> Can valence and arousal be reliably predicted from (1) the percentage of an image occupied by each of 20–25 color bins, (2) the image's top five dominant colors, and (3) its OASIS semantic category?

## Approach

1. **Color features** — Each image is analyzed pixel-by-pixel. Every pixel's hex code is mapped to one of 20–25 perceptual color bins. The result is a percentage-composition vector (one value per bin) plus the top five most dominant colors.
2. **Semantic features (Experiment 1)** — Use the existing OASIS category labels (Animals, Objects, Scenery, People) as a one-hot semantic feature.
3. **Semantic features (Experiment 2)** — Replace hand-labeled categories with predictions from a pretrained image classification model and measure whether this changes performance.
4. **Model** — A CNN (or regression baseline) trained on the combined feature vectors to predict valence and arousal, evaluated with K-fold cross validation. Performance is reported via log loss and visualized through composition charts.
5. **GUI** — A final interface that accepts any OASIS image and outputs a predicted valence/arousal score alongside a visualization of its color composition.

## Project Structure

```
oasis-emotion-prediction/
├── notebooks/
│   └── final_demo.ipynb       # Polished end-to-end demo (Colab-ready)
├── src/
│   ├── data_loader.py         # Load OASIS images and ratings CSV
│   ├── color_features.py      # Color bin composition + dominant color extraction
│   ├── semantic_features.py   # OASIS category encoding and classifier-based semantics
│   ├── model.py               # Model definition (CNN / regression baseline)
│   └── train.py               # K-fold training and evaluation pipeline
├── models/
│   └── saved_models/          # Serialized trained models
├── data/
│   └── oasis/                 # OASIS images and ratings CSV (not tracked in git)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the OASIS dataset from [benedekkurdi.com/#oasis](https://www.benedekkurdi.com/#oasis) or [https://db.tt/yYTZYCga](https://db.tt/yYTZYCga) and unzip/place the images and ratings CSV inside `data/oasis/`.

## Training

```bash
cd src
python train.py --csv ../data/oasis/OASIS.csv --images ../data/oasis/images/
```
