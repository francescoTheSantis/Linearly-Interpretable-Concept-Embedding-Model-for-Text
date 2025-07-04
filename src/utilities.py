import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
from omegaconf import DictConfig, OmegaConf, open_dict
from time import time
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from src.loaders.prompts import CEBAB_ZERO_SHOT, IMDB_ZERO_SHOT, \
    TREC_ZERO_SHOT, WOS_ZERO_SHOT, CLINC_ZERO_SHOT, BANK_ZERO_SHOT

# I used scienceplots for the style of the plots,
# but you can use any other style you want.
warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

def set_seed(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_loggers(cfg):
    """ Set the loggers for the experiment """
    # Update the note in the config: if it is None, set it to an empty string
    with open_dict(cfg):
        cfg.update(
            note = "_" if cfg.note is None else "_"+str(cfg.note)
        )
    name = f"{cfg.supervision}.{cfg.dataset.metadata.name}.{cfg.model.metadata.name}.{cfg.seed}.{int(time())}"
    group_format = (
        "{supervision}_"
        "{dataset}_"
        "{model}"
        "{note}"
    )
    # Define the tags for wandb
    tags = [cfg.dataset.metadata.name, cfg.model.metadata.name, cfg.supervision, cfg.note]
    # Define the group for wandb
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.wandb.project is None or cfg.wandb.entity is None:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(project=cfg.wandb.project, 
                               entity=cfg.wandb.entity, 
                               name=name,
                               group=group,
                               tags=tags)
        wandb_logger.log_hyperparams(parse_hyperparams(cfg))
    csv_logger = CSVLogger("logs/",
                           name="experiment_metrics")
    return wandb_logger, csv_logger

def parse_hyperparams(cfg: DictConfig):
    hyperparams = {
        "dataset": cfg.dataset.metadata.name,
        "model": cfg.model.metadata.name,
        "supervision": cfg.supervision,
        "seed": cfg.seed,
        "hydra_cfg": OmegaConf.to_container(cfg),
        "note": cfg.note,
    }
    return hyperparams


def check_concept_annotation(c):
    """
    Check if there are concept annotation and return a boolean value.
    The concepts annotations are not present if a column tensor of -1 is passed.
    If the concepts annotations are present, return True, otherwise return False.
    """
    if c[0,0] == -1:
        return False
    return True

def set_istruction_prompt(dataset_name: str) -> str:
    if dataset_name == 'banking':
        return BANK_ZERO_SHOT
    elif dataset_name == 'clinc':
        return CLINC_ZERO_SHOT
    elif dataset_name == 'cebab':
        return CEBAB_ZERO_SHOT
    elif dataset_name == 'imdb':
        return IMDB_ZERO_SHOT
    elif dataset_name == 'trec50':
        return TREC_ZERO_SHOT
    elif dataset_name == 'wos':
        return WOS_ZERO_SHOT
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for zero-shot LLM classification. Please provide a valid dataset name.")

def update_config_from_data(cfg: DictConfig, train_loader, c_names, y_names, c_groups, csv_log_dir) -> DictConfig:
    """
    Update the config with the input size, output size, and concept names.
    """
    ids, type, attention, embedding, c, y, gen_c = next(iter(train_loader))

    # If the concepts are not present, set c to None
    concept_annotations = check_concept_annotation(c)
    n_labels = len(y_names)

    if c_groups is None or not isinstance(c_groups, dict):
        c_groups = c_groups
    else:
        c_groups = dict(c_groups)
    
    with open_dict(cfg):
        cfg.engine.update(
            c_names = c_names,
            y_name = y_names,
            csv_log_dir = csv_log_dir,
            concept_annotations = concept_annotations,
            supervision = cfg.supervision,
        )
        
        if cfg.model.metadata.name in ['LLM_zero_shot', 'LLM_few_shot']:
            
            cfg.model.update(
                class_dict = {name: i for i, name in enumerate(y_names)},
                LLM = cfg.llm_name,
                storing_path = os.getcwd(),
                tokenizer = cfg.encoder_name,
                examples = cfg.dataset.metadata.examples,
                istruction_prompt = set_istruction_prompt(cfg.dataset.metadata.name),
            )
        else:
            cfg.model.params.update(
                output_size = n_labels,
                c_names = c_names,
                y_names = y_names,
                c_groups = c_groups,
                supervision = cfg.supervision,
            )
    return cfg


def plot_explanations(lmr_paths):

    for exp in lmr_paths:
        try:
            # If it does not exist, create the figs directory
            figs_path = f'figs/{exp['model']}_explanations/{exp['dataset']}'
            if not os.path.exists(figs_path):
                os.makedirs(figs_path)
            exp_info_path = os.path.join(exp['path'], 'logs/experiment_metrics/version_0')
            # Load all the required files
            pred_CBMs = torch.load(os.path.join(exp_info_path, 'pred_CBMs.pt'))
            c_trues = pd.read_csv(os.path.join(exp_info_path, 'c_trues.csv'))
            c_preds = pd.read_csv(os.path.join(exp_info_path, 'c_preds.csv'))
            y_trues = pd.read_csv(os.path.join(exp_info_path, 'y_trues.csv'))
            y_preds = pd.read_csv(os.path.join(exp_info_path, 'y_preds.csv'))
            binary_classification = True if len(y_trues.columns) == 2 else False

            # Sample N random samples from the test-set
            n_samples = 20
            random_indices = np.random.choice(len(y_trues), n_samples, replace=False)

            # Plot the explanations
            for i in range(n_samples):
                idx = random_indices[i]
                # Get the true and predicted concepts
                c_true = c_trues.iloc[idx].values
                c_pred = (c_preds.iloc[idx].values > 0.5).astype(int)

                # get the column name of the concepts corresponding to the highest value
                c_name_pred = [x.replace('_',' ') for idx, x in enumerate(c_preds.columns) if idx in np.argwhere(c_pred>0.5)]
                c_name_true = [x.replace('_',' ') for idx, x in enumerate(c_trues.columns) if idx in np.argwhere(c_true>0.5)]
                # Get the true and predicted labels
                if binary_classification:
                    y_true = y_trues.iloc[idx,1]
                    y_pred = y_preds.iloc[idx,1]
                else:
                    y_true = y_trues.iloc[idx].values.argmax(-1)
                    y_pred = y_preds.iloc[idx].values.argmax(-1)
                # get the column name of the concepts corresponding to the highest value
                y_name_pred = y_preds.columns[y_pred].replace('_',' ')
                y_name_true = y_trues.columns[y_true].replace('_',' ')
                # Get the weights associated to the predicted class of the 
                # predicted CBM
                if binary_classification:
                    y_pred = 0
                pred_CBM = pred_CBMs[idx,y_pred,:,:].squeeze().cpu().numpy()
                # Perform the element-wise multiplication
                logits = np.multiply(pred_CBM, c_pred)

                # Sort the logits by absolute value in ascending order and take the k highest values
                top_k = 10
                indices = np.argsort(np.abs(logits))[::-1][:top_k]
                logits = logits[indices]
                c_preds_names = [c_preds.columns[idx] for idx in indices]
                # c_trues_names = [c_trues.columns[idx] for idx in indices]

                # Plot the explanations
                fig, ax = plt.subplots(figsize=(10, 5))
                bar_colors = ['tab:blue' if val >= 0 else 'tab:red' for val in logits]
                y_pos = np.arange(len(logits))
                ax.barh(
                    y_pos, logits, color=bar_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.6
                )
                # Add black vertical line at 0
                ax.axvline(x=0, color='black', linewidth=0.9)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([x.replace('_',' ') for x in c_preds_names], 
                                    fontsize=14)
                ax.set_xlabel('Logit Value', fontsize=12)
                ax.set_title(f'Predicted class: {y_name_pred}, True class: {y_name_true}', 
                            fontsize=12)
                # Eliminate minor ticks
                ax.xaxis.set_minor_locator(plt.NullLocator())
                ax.yaxis.set_minor_locator(plt.NullLocator())
                # Save the figure
                plt.tight_layout()
                plt.savefig(f'{figs_path}/explanations_{i}.pdf')
        except:
            print(f"Error while plotting explanations for {exp['model']} on {exp['dataset']}. Skipping...")
            continue
    print("Explanations plotted successfully.")


def plot_sparsity_ablation(performance, sparsity_threshold=1e-5, output_dir='figs', style=None):
    """
    Plots task accuracy vs concept sparsity for each supervision strategy and dataset.
    Averages results over seeds, includes standard deviation shading, and supports custom styling.
    """
    
    default_style = {
        'title_size': 14,
        'label_size': 12,
        'tick_size': 10,
        'xtick_labelsize': 16,
        'ytick_labelsize': 16,
        'legend_fontsize': 10,
        'line_width': 2,
        'marker_size': 6,
        'colors': ['tab:red', 'tab:green', 'tab:blue'],  # List of colors, or None to auto-generate
        'alpha': 0.3     # Transparency for standard deviation shading
    }
    if style is None:
        style = {}
    style = {**default_style, **style}

    os.makedirs(output_dir, exist_ok=True)
    grouped = performance.groupby('supervision')

    for supervision, group in grouped:
        datasets = group['dataset'].unique()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        used_labels = set()

        for i, dataset in enumerate(datasets):
            dataset_group = group[group['dataset'] == dataset]
            grouped_by_reg = dataset_group.groupby('weight_reg')

            weight_regs = []
            mean_accuracies, std_accuracies = [], []
            mean_sparsities, std_sparsities = [], []

            for weight_reg, sub_group in grouped_by_reg:
                weight_regs.append(weight_reg)

                # Task accuracy mean and std
                task_vals = sub_group['task'].values
                mean_accuracies.append(np.mean(task_vals))
                std_accuracies.append(np.std(task_vals))

                # Concept sparsity mean and std
                sparsities = []
                for _, row in sub_group.iterrows():
                    c_preds = row['c_preds'].values
                    pred_weights = row['pred_weights']
                    weighted_preds = c_preds[:, :, None] * pred_weights
                    mask = np.abs(weighted_preds) > sparsity_threshold
                    sparsities.append(np.mean(mask))
                mean_sparsities.append(np.mean(sparsities))
                std_sparsities.append(np.std(sparsities))

            # Sort by weight_reg
            sorted_idx = np.argsort(weight_regs)
            weight_regs = np.array(weight_regs)[sorted_idx]
            mean_accuracies = np.array(mean_accuracies)[sorted_idx]
            std_accuracies = np.array(std_accuracies)[sorted_idx]
            mean_sparsities = np.array(mean_sparsities)[sorted_idx]
            std_sparsities = np.array(std_sparsities)[sorted_idx]

            label = dataset if dataset not in used_labels else None
            color = (style['colors'][i] if style['colors'] and i < len(style['colors']) else None)

            # Task accuracy plot
            line1, = axes[0].plot(weight_regs, mean_accuracies, label=label, marker='o',
                                  linewidth=style['line_width'], markersize=style['marker_size'], color=color)
            color1 = line1.get_color()
            axes[0].fill_between(weight_regs, mean_accuracies - std_accuracies,
                                 mean_accuracies + std_accuracies, color=color1, alpha=style['alpha'])

            # Concept sparsity plot
            line2, = axes[1].plot(weight_regs, mean_sparsities, label=label, marker='o',
                                  linewidth=style['line_width'], markersize=style['marker_size'], color=color)
            color2 = line2.get_color()
            axes[1].fill_between(weight_regs, mean_sparsities - std_sparsities,
                                 mean_sparsities + std_sparsities, color=color2, alpha=style['alpha'])

            used_labels.add(dataset)

        # Style axes
        for ax, ylabel, title in zip(
            axes,
            ['Task Accuracy', 'Concept Sparsity'],
            [f'Task Accuracy vs Weight Regularization ({supervision})',
             f'Concept Sparsity vs Weight Regularization ({supervision})']
        ):
            ax.set_xlabel('Weight Regularization', fontsize=style['label_size'])
            ax.set_ylabel(ylabel, fontsize=style['label_size'])
            #ax.set_title(title, fontsize=style['title_size'])
            ax.set_xscale('log')
            ax.grid(True)

            # Remove all minor ticks on all sides
            ax.tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)
            ax.tick_params(labelsize=style['tick_size'])

        # Unified legend below the figure
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                   loc='lower center', ncol=len(by_label),
                   bbox_to_anchor=(0.5, -0.05),
                   frameon=True,  # Enable the legend frame
                   facecolor='white',  # Set the background color
                   edgecolor='tab:grey',  # Set the border color
                   fontsize=style['legend_fontsize'])

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(output_dir, f'sparsity_ablation_{supervision}.png'), bbox_inches='tight')
        plt.show()
