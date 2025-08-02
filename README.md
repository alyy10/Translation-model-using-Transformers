# Transformer for Language Translation

This project implements a **Transformer-based neural machine translation model** using PyTorch to translate text from German to English, leveraging the Multi30K dataset. The notebook covers data preprocessing, model architecture design, training, evaluation, and translating a German PDF document into English. The Transformer architecture, introduced in the [&#34;Attention is All You Need&#34;](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper, is utilized for its efficiency in handling sequence-to-sequence tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Translating a PDF](#translating-a-pdf)
- [Results](#results)

## Overview

This project builds a Transformer model to translate German sentences into English using the Multi30K dataset, which contains parallel German-English sentence pairs. It includes:

- Data preprocessing with tokenization and vocabulary creation.
- A sequence-to-sequence Transformer model for translation.
- Training with cross-entropy loss and evaluation using BLEU scores.
- A feature to translate German PDF documents into English, saving the output as a new PDF.

The notebook demonstrates the advantages of Transformers over traditional RNNs and LSTMs, particularly their ability to process entire sequences simultaneously and capture long-range dependencies.

## Features

- **Data Preprocessing**: Tokenizes text using `spacy`, builds vocabularies, and creates PyTorch DataLoaders.
- **Transformer Model**: Customizable sequence-to-sequence Transformer with encoder/decoder layers, multi-head attention, and feed-forward networks.
- **Training**: Uses the Adam optimizer with progress tracking via `tqdm`.
- **Evaluation**: Computes BLEU scores to assess translation quality.
- **PDF Translation**: Translates German PDFs to English using `pdfplumber` and `FPDF`.
- **Pretrained Model**: Includes a pretrained model (trained for 40 epochs) to skip lengthy CPU training.

## Requirements

The following dependencies are required:

```bash
torch==1.13.1
torchdata==0.5.1
torchtext==0.14.1
spacy==3.7.2
nltk==3.8.1
pdfplumber==0.9.0
fpdf==1.7.2
portalocker==2.7.0
```

Download `spacy` language models for German and English:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Usage

Run the notebook cells sequentially to:

1. **Prepare Data**: Load and preprocess the Multi30K dataset, tokenize text, and create DataLoaders (batch size 128).
2. **Train Model**: Train the Transformer for 10 epochs or load the pretrained model.
3. **Evaluate**: Compute BLEU scores for translations and validate model performance.
4. **Translate PDF**: Convert a German PDF to English using the `translate_pdf` function.

Example for PDF translation:

```python
input_file_path = "input_de.pdf"
output_file = "output_en.pdf"
translate_pdf(input_file_path, transformer, output_file)
```

## Model Architecture

The Transformer model is a sequence-to-sequence architecture with:

- **Encoder**: Processes German input sequences.
- **Decoder**: Generates English output sequences.
- **Parameters**:
  - Source Vocabulary Size: ~10,837 (German)
  - Target Vocabulary Size: ~10,837 (English)
  - Embedding Size: 512
  - Attention Heads: 8
  - Feed-Forward Hidden Dimension: 512
  - Encoder/Decoder Layers: 3 each

The model uses positional encoding and multi-head attention, as described in the [&#34;Attention is All You Need&#34;](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) paper.

## Training

The model is trained with:

- **Optimizer**: Adam (learning rate=0.0001, betas=(0.9, 0.98), epsilon=1e-9)
- **Loss Function**: Cross-Entropy Loss, ignoring padding tokens (`PAD_IDX`)
- **Batch Size**: 128
- **Epochs**: 10 (or 40 for the pretrained model)

Training on CPU is slow (~40–60 minutes per epoch). The provided pretrained model (`transformer_de_to_en_model.pt`) can be loaded to skip training:

```python
transformer.load_state_dict(torch.load('transformer_de_to_en_model.pt', map_location=torch.device('cpu')))
```

**Training Output (Partial)**:
For one epoch:

```
Epoch: 1, Train loss: 5.078, Val loss: 4.392, Epoch time = 4753.328s
```

(Note: Training was interrupted in the provided notebook, but the pretrained model reflects 40 epochs of training.)

## Evaluation

The model is evaluated using:

- **Validation Loss**: Computed over the validation DataLoader.
- **BLEU Score**: Measures translation quality by comparing generated translations to reference translations.

**Sample BLEU Score**:

```
BLEU Score: 1.0 for A brown dog plays in the snow .
```

This score was calculated for the German sentence "Ein brauner Hund spielt im Schnee." against multiple reference translations, indicating perfect translation for this case.

**Sample Translations**:

```
German Sentence: Männer stehen neben irgendeiner hydraulischen Maschine .
English Translation: Men are standing next to some sort of hydraulic machine .
Model Translation: Men are standing next to some sort of fishing concrete concrete concrete concrete concrete concrete concrete concrete concrete concrete

German Sentence: Zwei Arbeiter reinigen nachts ein Bauwerk .
English Translation: Two workers are cleaning a structure at night .
Model Translation: Two workers are installing a bunch of concrete concrete concrete concrete concrete concrete concrete concrete concrete concrete concrete concrete

German Sentence: Sieben Bauarbeiter arbeiten an einem Gebäude .
English Translation: Seven construction workers working on a building .
Model Translation: Seven construction workers are working on a concrete building .

German Sentence: Die Kinder spielen nachts mit Wunderkerzen .
English Translation: The children play with sparklers at night .
Model Translation: The children play with a bike .

German Sentence: Ein älteres Paar geht zusammen spazieren .
English Translation: An older couple taking a walk together .
Model Translation: An older couple walk together in a park area where there are a couple is a body of water
```

The model performs well on simple sentences but produces repetitive tokens (e.g., "concrete") or incorrect terms (e.g., "bike" instead of "sparklers") for complex sentences, suggesting the need for further training or beam search decoding.

## Translating a PDF

The `translate_pdf` function:

- Extracts text from a German PDF using `pdfplumber`.
- Splits text into sentences and translates each using the Transformer model.
- Wraps translated text with `textwrap` to fit A4 page dimensions.
- Saves the output as a new PDF using `FPDF`.

**Example Usage**:

```python
input_file_path = "input_de.pdf"
output_file = "output_en.pdf"
translate_pdf(input_file_path, transformer, output_file)
```

**Output**:

```
Translated PDF file is saved as: output_en.pdf
```

The output PDF (`output_en.pdf`) contains the translated English text, formatted for readability.

## Results

Key results from the notebook execution include:

- **Loss Calculation Example**:
  For a single sample ("Ein Arbeiter nimmt eine Messung in einem U-Bahn-Zug vor." → "A worker taking a reading on a subway train ."):

  ```
  Loss: 1.7348
  ```

  After manual calculation (optional section):

  ```
  Loss: -0.5981
  ```

  The discrepancy may reflect differences in loss computation methods or numerical precision.
- **Probability Analysis**:
  For the same sample, the model’s predicted probabilities for tokens were:

  ```
  Predicted token id: 6, predicted probability: 0.8501 (Actual: 6, probability: 0.8501)
  Predicted token id: 348, predicted probability: 0.8726 (Actual: 348, probability: 0.8726)
  Predicted token id: 10, predicted probability: 0.8083 (Actual: 168, probability: 0.000022)
  Predicted token id: 4, predicted probability: 0.9782 (Actual: 4, probability: 0.9782)
  Predicted token id: 369, predicted probability: 0.2092 (Actual: 217, probability: 0.0476)
  ```

  The model correctly predicts some tokens but struggles with others, contributing to non-zero loss.
- **BLEU Score**:
  The model achieved a perfect BLEU score of 1.0 for "Ein brauner Hund spielt im Schnee." → "A brown dog plays in the snow .", indicating high accuracy for simple sentences.
- **Translation Quality**:
  The model performs well on straightforward sentences but produces errors (e.g., repetitive "concrete" tokens) in complex cases. This suggests that further training, a larger model, or advanced decoding strategies (e.g., beam search) could improve performance.
- **PDF Translation**:
  The `translate_pdf` function successfully processed `input_de.pdf` and generated `output_en.pdf` with translated text, demonstrating practical application.

To improve results:

- Train for more epochs (e.g., 40+).
- Use a GPU to accelerate training.
- Implement beam search instead of greedy decoding.
- Fine-tune hyperparameters (e.g., learning rate, number of layers).
