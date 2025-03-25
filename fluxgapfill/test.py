import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

from fluxgapfill.models import EnsembleModel
from fluxgapfill.metrics import metric_dict, uncertainty_metric_dict
from fluxgapfill.models import get_model_class


def test(
        data_dir,
        df,
        models,
        distribution='laplace',
        test_metrics=list(metric_dict.keys()),
        uncertainty_test_metrics=list(uncertainty_metric_dict.keys())
):
    '''
    Evaluate models on the test set.

    Args:
        data_dir (str): Site-specific directory for all data
        df (pd.DataFrame): Dataframe object containing all relevant site data for
                           modelling. Assumes Ameriflux formatting.
        models (list<str>): Comma-separated list of model names to train.
                            Options: ['rf', 'ann', 'lasso', 'xgb']
        predictors (list<str>): Comma-separated list of predictors. Ignored if
                                predictors_path is provided.
                                Certain keyword predictors are used to denote
                                specific sets of predictors:
                                ['temporal', 'all']
        distribution (str): Which distribution to use for prediction.
                            Options: ['laplace', 'normal']
        test_metrics (list<str>): Metrics to use to evaluate the model(s) on
                                  the split.
        uncertainty_test_metrics (list<str>): Metrics to use to evaluate the
                                              uncertainty estimates of the
                                              model(s) on the test set.
        overwrite (bool): Whether to overwrite results if they already exist.
    '''
    for model in models:
        try:
            get_model_class(model)
        except Exception as e:
            raise ValueError(f'Model {model} not supported.')
    
    data_dir = Path(data_dir)
    df = df.copy().set_index('TIMESTAMP_START')
    
    models_path = data_dir / 'models'
    index_path = data_dir / 'indices'
    test_idx = np.load(index_path / 'test.npy')
    df_test = df.loc[test_idx]
    for model in models:
        model_path = models_path / model

        Model = EnsembleModel(model_path)
        y = df_test['FCH4']
        y_hat_dist = Model.predict_dist(
            df_test,
            distribution=distribution
        )
        y_hat = [dist.mean() for dist in y_hat_dist]
        y_hat_scale = [dist.scale for dist in y_hat_dist]
        uncertainty_scale = Model.uncertainty_scale(
            y, y_hat_dist, distribution=distribution
        )
        y_hat_dist_scaled = Model.predict_dist(
            df_test,
            distribution=distribution,
            uncertainty_scale=uncertainty_scale
        )
        y_hat_scale_scaled = [dist.scale for dist in y_hat_dist_scaled]
        df_pred = pd.DataFrame({
            'groundtruth': y,
            'prediction': y_hat,
            'uncertainty_scale': y_hat_scale,
            'scaled_uncertainty_scale': y_hat_scale_scaled
        })
        
        df_pred.to_csv(model_path / 'test_predictions.csv', index=False)
        
        scale_path = model_path / 'scale.json'
        with scale_path.open('w') as f:
            json.dump(uncertainty_scale, f)

        scores = {
            eval_metric: [metric_dict[eval_metric](y, y_hat)]
            for eval_metric in test_metrics
        }
        uncertainty_scores = {
            unc_eval_metric: [
                uncertainty_metric_dict[unc_eval_metric](y, y_hat_dist)
            ]
            for unc_eval_metric in uncertainty_test_metrics
        }
        scaled_uncertainty_scores = {
            f'{unc_eval_metric}_scaled': [
                uncertainty_metric_dict[unc_eval_metric](
                    y, y_hat_dist_scaled
                )
            ]
            for unc_eval_metric in uncertainty_test_metrics
        }
        scores = {
            **scores,
            **uncertainty_scores,
            **scaled_uncertainty_scores
        }

        scores_flipped = defaultdict(list)
        for k, v in scores.items():
            scores_flipped['metric'].append(k)
            scores_flipped['value'].append(v[0])
        
        # format output
        df_scores = pd.DataFrame(scores_flipped)
        df_scores.to_csv(model_path / 'test_metrics.csv', index=False)
