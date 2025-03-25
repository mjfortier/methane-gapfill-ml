import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from fluxgapfill.models import get_model_class


def is_trained(model_dir, num_splits):
    if not os.path.exists(model_dir):
        return False
    
    model_name = model_dir.stem
    for i in range(num_splits):
        if not os.path.exists(model_dir / f'{model_name}{i}.pkl'):
            return False
    
    if not os.path.exists(model_dir / 'val_metrics.csv'):
        return False
    return True


def train(
        data_dir,
        df,
        models,
        predictors,
        inner_cv=5,
        n_iter=20,
        log_metrics=['pr2', 'nmae'],
        overwrite=False
):
    '''
    Train models using predictors. This function assumes there are
    no stale models. If overwrite is False and models are detected
    in the directory, this function will not retrain them.

    Args:
        data_dir (str): Site-specific directory for all data
        df (pd.DataFrame): Dataframe object containing all relevant site data for
                           modelling. Assumes Ameriflux formatting.
        models (list<str>): Comma-separated list of model names to train.
                            Options: ['rf', 'ann', 'lasso', 'xgb']
        predictors (list<str>): List of predictors to use. If 'all' is given,
                                then all predictors in the dataframe will be used.
        inner_cv (int): Number of folds for k-fold cross validation in the
                        training set(s) for selecting model hyperparameters.
        n_iter (int): Number of parameter settings that are sampled in the
                      inner cross validation.
        log_metrics (list<str>): Validation metrics to log.
        overwrite (bool): Whether to overwrite models if they already exist.
    '''
    for model in models:
        try:
            get_model_class(model)
        except Exception as _:
            raise ValueError(f'Model {model} not supported.')
    
    data_dir = Path(data_dir)
    df = df.copy().set_index('TIMESTAMP_START')

    models_path = data_dir / 'models'
    index_path = data_dir / 'indices'
    if not os.path.exists(index_path):
        raise ValueError('Preprocessing must be run before model training.')
    
    num_splits = len(list(index_path.glob('train*.npy')))
    if num_splits < 1:
        raise ValueError('Must have at least one set of train / val indices')
    train_val_splits = [(np.load(index_path / f'train{i}.npy'), np.load(index_path / f'val{i}.npy'))
                        for i in range(num_splits)]
    
    for model in models:
        model_path = models_path / model
        if is_trained(model_path, num_splits) and not overwrite:
            print(f'Model {model} is already trained. Skipping...')
            continue

        if os.path.exists(model_path):
            os.removedirs(model_path)
        os.makedirs(model_path, exist_ok=True)

        ModelClass = get_model_class(model)
        all_scores = defaultdict(list)
        print(f'Training model {model}...')
        for i, train_idx, val_idx in tqdm(enumerate(train_val_splits)):
            df_train = df.loc[train_idx]
            df_val = df.loc[val_idx]

            model_obj = ModelClass(predictors=predictors,
                                            cv=inner_cv, n_iter=n_iter)
            model_obj.fit(df_train, 'FCH4')
            model_scores = model_obj.evaluate(df_val, 'FCH4', log_metrics)
            for score, metric in zip(model_scores, log_metrics):
                all_scores[metric].append(score)
            
            model_obj.save(model_path / f'{model}{i}.pkl')
        
        scores_df = pd.DataFrame(all_scores)
        scores_df.to_csv(model_path / 'val_metrics.csv', index=False)
