# Linearly Interpretable Concept Embedding Model for Text (LICEM)

This repository contains the implementation of the [Linearly Interpretable Concept Embedding Model (LICEM)](https://arxiv.org/pdf/2406.14335) for text classification tasks. LICEM leverages concept-based embeddings to provide interpretable predictions while maintaining competitive performance.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Logging and Results](#logging-and-results)
6. [License](#license)

---

## Installation

To set up the environment, follow these steps:

1. Clone the repository

2. Move to the working directory:
   ```bash
   cd Linearly-Interpretable-Concept-Embedding-Model-for-Text
   ```

3. Install dependencies using `conda`:
   ```bash
   conda env create -f environment.yml
   conda activate licem
   ```

4. Configure environment variables:
   After installing dependencies, update the environment variables in `env.py`:
   - **PROJECT_NAME**: Set this to your project name.
   - **HOME**: Specify the path to your project directory.
   - **OPENAI_API_KEY**: Replace this with your OpenAI API key.
   - **MISTRALAIKEY**: Replace this with your Mistral AI API key.

   These variables are essential for the proper functioning of the code.

---

## Dataset Preparation

Ensure your datasets are stored in the directory specified in `env.py`.

Supported datasets include:
- `cebab`
- `imdb`
- `trec50`
- `wos`
- `clinc`
- `banking`

If you want to use preprocessed datasets, set `use_stored_dataset: True` in the configuration file (`conf/general_sweep.yaml`).

---

## Configuration

The experiments are configured using Hydra. The main configuration file is located at `conf/general_sweep.yaml`. Key parameters include:

- **Dataset**: Specify the dataset to use (`dataset`).
- **Model**: Choose the model (`model`).
- **Supervision Strategy**: Define the supervision strategy (`supervision`).
- **Training Parameters**: Adjust parameters like `max_epochs`, `batch_size`, and `lr_patience`.

For detailed configuration options, refer to the comments in `conf/general_sweep.yaml`.

---

## Running Experiments

To reproduce the experiments, run the following command:

```bash
python main.py
```

This script will:
1. Load or preprocess the dataset.
2. Train the specified model.
3. Evaluate the model on the test set.

This will repeated for each combination of parameters defined in the `sweeper` section of `conf/general_sweep.yaml`.

---

## Logging and Results

### WandB Integration

This repository supports logging with [Weights & Biases](https://wandb.ai). To enable WandB logging:
1. Set your WandB project name and entity in `conf/general_sweep.yaml`:
   ```yaml
   wandb:
     project: "licem"
     entity: "your-wandb-entity"
   ```

2. Results will be logged to your WandB dashboard.

### Local Results

Results, including intervention data and metrics, are stored in the `outputs` directory. The path is dynamically generated based on the current timestamp.

---

## Citation

If you use LICEM in your research, please cite our work:

```bibtex
@article{de2024self,
  title={Self-supervised Interpretable Concept-based Models for Text Classification},
  author={De Santis, Francesco and Bich, Philippe and Ciravegna, Gabriele and Barbiero, Pietro and Giordano, Danilo and Cerquitelli, Tania},
  journal={arXiv preprint arXiv:2406.14335},
  year={2024}
}
```
