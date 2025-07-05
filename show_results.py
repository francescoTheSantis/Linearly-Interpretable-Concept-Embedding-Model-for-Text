import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import warnings
import os
import yaml
from matplotlib.ticker import FuncFormatter
from src.utilities import plot_explanations, plot_sparsity_ablation, plot_sparsity_vs_accuracy
import torch

# I used scienceplots for the style of the plots, but you can use any other style you want.
warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

# List the paths containing the results
paths = [
    "", # Path to the first set of results
    # Add more paths if needed
]

# create the figs folder if it does not exist
os.makedirs('figs', exist_ok=True)

###### Collect results regarding concept/task performance ######

exps_path = []
lmr_paths = []
for path in paths:
    exps = os.listdir(path)
    exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

performance = pd.DataFrame()

for exp in exps_path:
    d = {}
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv')  
    llm_result_file = os.path.join(exp, 'results.csv')
    #if os.path.exists(conf_file) and (os.path.exists(result_file) or os.path.exists(llm_result_file)):
    try:
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)
        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        d['model'] = conf['model']['metadata']['name']

        if d['model'] in ['LLM_zero_shot', 'LLM_few_shot']:
            llm_acc = pd.read_csv(llm_result_file, header=0)['accuracy'].iloc[0]
            d['task'] = llm_acc
            d['concept'] = 0
            d['supervision'] = 'self-generative' # LLM do not follow any of the learning paradigms, we set it to self-generative as default.
        else:
            d['supervision'] = conf['supervision']

            with open(result_file, 'r') as file:
                result = pd.read_csv(file, header=0)

            d['task'] = result['test_task_acc'].iloc[-1]

            # Select the last row of the dataframe where we test the model
            try:
                d['concept'] = result['test_concept_acc'].iloc[-1]
            except KeyError:
                d['concept'] = 0

            #print(d)
            
            # if the model is licem, we need to collect the explanations
            if d['model'] == 'licem' and d['seed']==1:
                expl_dict = d.copy()
                expl_dict['path'] = exp
                lmr_paths.append(expl_dict)

        performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
    except Exception as e:
        print(f"Error processing {exp}: {e}")


######### Dataset and model styles #########

def get_df_name(df):
    if df=='imdb':
        return 'IMDB'
    elif df=='cebab':
        return 'CEBaB'
    elif df=='trec50':
        return 'TREC-50'
    elif df=='wos':
        return 'WOS'
    elif df=='clinc':
        return 'CLINC-OOS'
    elif df=='banking':
        return 'Banking-77'

marker_size = 14

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
model_styles = {
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:orange', 'size': marker_size},
    'cbm_linear': {'marker': '*', 'name': 'CBM+Linear', 'color': 'tab:olive', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:red', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:purple', 'size': marker_size},
    'cbm_dt': {'marker': 'v', 'name': 'CBM+DT', 'color': 'tab:pink', 'size': marker_size},
    'cbm_xg': {'marker': 'v', 'name': 'CBM+XG', 'color': 'tab:green', 'size': marker_size},
    'dcr': {'marker': 'h', 'name': 'DCR', 'color': 'tab:gray', 'size': marker_size},
    'LLM_zero_shot': {'marker': 'D', 'name': 'LLM Zero-Shot', 'color': 'tab:cyan', 'size': marker_size},
    'LLM_few_shot': {'marker': 'D', 'name': 'LLM Few-Shot', 'color': 'tab:cyan', 'size': marker_size},
    'licem': {'marker': 's', 'name': 'LICEM', 'color': 'tab:blue', 'size': marker_size},
}

# Define the custom order
# If the experiment you run does not contain a dataset, just remove it from the list.
custom_order = [#'cebab',
                'imdb',
                'trec50',
                #'wos',
                'clinc',
                'banking'
                ]


# Filter the performance dataframe to keep only the models in model_styles 
# and datasets in custom_order.
performance = performance[performance['model'].isin(model_styles.keys()) & \
                          performance['dataset'].isin(custom_order)]





####################################################################################
############################ Plot explanations #####################################
####################################################################################

plot_explanations(lmr_paths)





#############################################
########## Task & Concept Accuracy ##########
#############################################

# Filter data for 'task' and 'concept'
task_df = performance.copy()
task_df = task_df.rename(columns={'task': 'accuracy'})
concept_df = performance.copy()
concept_df = concept_df.rename(columns={'concept': 'accuracy'})

# Compute mean and std for 'task'
task_stats = task_df.groupby(['model', 'dataset', 'supervision']).agg(
    avg_accuracy_task=('accuracy', 'mean'),
    std_accuracy_task=('accuracy', 'std')
).reset_index().fillna(0)

# Compute mean and std for 'concept'
concept_stats = concept_df.groupby(['model', 'dataset', 'supervision']).agg(
    avg_accuracy_concept=('accuracy', 'mean'),
    std_accuracy_concept=('accuracy', 'std')
).reset_index().fillna(0)

########## Task Accuracy Table ##########

task_avg = task_stats[['model', 'dataset', 'supervision', 'avg_accuracy_task']]
task_std = task_stats[['model', 'dataset', 'supervision', 'std_accuracy_task']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset', 'supervision'])

# Print for each supervision strategy 
for supervision in merged_task['supervision'].unique():
    filtered_task_avg = task_avg[task_avg['supervision'] == supervision]
    filtered_task_std = task_std[task_std['supervision'] == supervision]
    # Create a pivot table with the desired format
    pivot_table_avg = filtered_task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_task'])
    pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
    pivot_table_std = filtered_task_std.pivot(index='model', columns='dataset', values=['std_accuracy_task'])
    pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

    final_table = pd.DataFrame()
    for i, row in pivot_table_avg.iterrows():
        d={}
        for j in pivot_table_std.columns:
            acc = row[j]*100
            std = pivot_table_std.loc[i, j]*100
            d[j] = f"{acc:.2f} ± {std:.2f}"
        # add a column to the final_table dataframe called row.name which contains d
        final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
        
    # Reindex the columns of final_table according to the custom order
    final_table = final_table.reindex(columns=custom_order)

    # Replace the model and dataset names
    final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)
    final_table.columns = final_table.columns.map(lambda x: get_df_name(x))

    print('\n\n')
    print(f'Task Accuracy with supervision strategy: {supervision}')
    print('-------------------')
    print(final_table)

    # store the table in a csv file
    final_table.to_csv(f'figs/task_accuracy_{supervision}.csv', index=True)

########## Concept Accuracy Table ##########

concept_avg = concept_stats[['model', 'dataset', 'supervision', 'avg_accuracy_concept']]
concept_std = concept_stats[['model', 'dataset', 'supervision', 'std_accuracy_concept']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_concept = pd.merge(concept_avg, concept_std, on=['model', 'dataset', 'supervision'])

for supervision in merged_task['supervision'].unique():
    filtered_concept_avg = concept_avg[task_avg['supervision'] == supervision]
    filtered_concept_std = concept_std[task_std['supervision'] == supervision]
        
    # Create a pivot table with the desired format
    pivot_table_avg = filtered_concept_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_concept'])
    pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
    pivot_table_std = filtered_concept_std.pivot(index='model', columns='dataset', values=['std_accuracy_concept'])
    pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

    final_table = pd.DataFrame()
    for i, row in pivot_table_avg.iterrows():
        d={}
        for j in pivot_table_std.columns:
            acc = row[j]*100
            std = pivot_table_std.loc[i, j]*100
            d[j] = f"{acc:.2f} ± {std:.2f}"
        # add a column to the final_table dataframe called row.name which contains d
        final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
        
    # Reindex the columns of final_table according to the custom order
    final_table = final_table.reindex(columns=custom_order)

    # Replace the model and dataset names
    final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)
    final_table.columns = final_table.columns.map(lambda x: get_df_name(x))

    print('\n\n')
    print(f'Concept Accuracy with supervision strategy: {supervision}')
    print('-------------------')
    print(final_table)

    # store the table in a csv file
    final_table.to_csv(f'figs/{supervision}_concept_accuracy.csv', index=True)


#######################################
########## Sparsity ablation ##########
#######################################

paths = [
    "", # Path to the first set of results
    # Add more paths if needed
]

###### Collect results regarding task performance and concept sparsity ######

exps_path = []
for path in paths:
    exps = os.listdir(path)
    exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

performance = pd.DataFrame()

for exp in exps_path:
    d = {}
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv')  
    llm_result_file = os.path.join(exp, 'results.csv')
    c_preds_file = os.path.join(exp, 'logs/experiment_metrics/version_0/c_preds.csv')
    c_trues_file = os.path.join(exp, 'logs/experiment_metrics/version_0/c_trues.csv')
    ids_file = os.path.join(exp, 'logs/experiment_metrics/version_0/ids.pt')
    pred_weights_file = os.path.join(exp, 'logs/experiment_metrics/version_0/pred_weights.pt')
    try:
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)
        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        model = conf['model']['metadata']['name']

        if model in ['licem', 'cbm_linear']:
            d['supervision'] = conf['supervision']
            d['model'] = model
            with open(result_file, 'r') as file:
                result = pd.read_csv(file, header=0)

            d['task'] = result['test_task_acc'].iloc[-1]

            # Select the last row of the dataframe where we test the model
            try:
                d['concept'] = result['test_concept_acc'].iloc[-1]
            except KeyError:
                d['concept'] = 0

            # get weight regularization
            d['weight_reg'] = conf['weight_reg']

            # Read c_preds and c_trues
            c_preds = pd.read_csv(c_preds_file, header=0)
            c_trues = pd.read_csv(c_trues_file, header=0)
            # read ids, which is a torch tensor
            ids = torch.load(ids_file).cpu().numpy()
            # read pred_weights, which is a torch tensor
            pred_weights = torch.load(pred_weights_file).squeeze().cpu().numpy()
            d['c_preds'] = c_preds
            d['c_trues'] = c_trues
            d['ids'] = ids
            d['pred_weights'] = pred_weights

            performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
    except Exception as e:
        print(f"Error processing {exp}: {e}")


dataset_to_show = ['trec50']  # Specify the datasets you want to plot

# Plot the sparsity ablation results
plot_sparsity_ablation(performance, dataset_func=get_df_name, datasets_to_plot=dataset_to_show)

# Plot sparsity vs task accuracy
plot_sparsity_vs_accuracy(performance, dataset_func=get_df_name, datasets_to_plot=dataset_to_show)