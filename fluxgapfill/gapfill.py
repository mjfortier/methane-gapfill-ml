import os
import json
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta

from fluxgapfill.models import EnsembleModel
from fluxgapfill.metrics import get_pred_interval
from fluxgapfill.models import get_model_class

def gapfill(
        data_dir,
        df,
        models,
        distribution="laplace"
):
    """
    Gapfill data with trained models.

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
        predictors_paths (list<str>): Comma-separated list of paths to files
                                      containing predictor names. See
                                      predictors/metereological.txt for an
                                      example.
        distribution (str): Which distribution to use for prediction.
                            Options: ['laplace', 'normal']
        budget_date_ranges_path (str): Dictionary mapping site names to date
                                       ranges. See
                                           gapfill/budget_date_ranges.json
                                       for an example.

    Writes gapfilled data to
        data/{SiteID}/gapfilled/{model}_{predictors}_{distribution}.csv
    where the CSV has all the same columns as raw.csv, excluding any existing
    gapfilled columns, plus columns for the
        predicted mean (FCH4_F)
        95% uncertainty (FCH4_uncertainty)
        spread individual predictions (FCH4{1-N})
    and writes budget data to
        data/{SiteID}/gapfilled/{model}_{predictors}_{distribution}_budget.csv
    """
    # parse inputs
    data_dir = Path(data_dir)
    df = df.copy().set_index('TIMESTAMP_START')
        
    models_dir = data_dir / 'models'
    if models is None:
        models = os.listdir(models_dir)
    
    for model in models:
        model_dir = models_dir / model
        
        print(f'Gapfilling: {model}, {distribution}')
        df_gapfilled = gapfill_site_model_predictor(model_dir, df, distribution)
        df_gapfilled.to_csv(model_dir / f"gapfilled_{distribution}.csv", index=False)
        
        print(f'Computing Annual Budget: {model}, {distribution}')
        # For budget estimates, convert from nmol m-2 s-1 to g C m-2 halfhour-1
        ch4_conversion = lambda fch4: fch4*60*30*12.0107*10**-9
        for col in [col for col in df_gapfilled.columns if 'FCH4' in col]:
            df_gapfilled[col] = df_gapfilled[col].apply(ch4_conversion)
        
        df_budget = compute_annual_budget(df_gapfilled)
        df_budget.to_csv(model_dir / f"gapfilled_{distribution}_budget.csv", index=False)


def gapfill_site_model_predictor(model_dir, df, distribution):
    Model = EnsembleModel(model_dir)
    scale_file = model_dir / "scale.json"
    if not scale_file.exists():
        raise ValueError(
            "Must run <python main.py test> with --distribution " +
            f"{distribution} to compute an uncertainty scale before " +
            "gapfilling."
        )
    with scale_file.open() as f:
        uncertainty_scale = json.load(f)

    y_hat_individual = Model.predict_individual(df)
    y_hat_mean = np.mean(y_hat_individual, axis=0)
    y_hat_spread = (
        (y_hat_individual - y_hat_mean) * uncertainty_scale + y_hat_mean
    )
    y_hat_dist = Model.predict_dist(
        df,
        distribution=distribution,
        uncertainty_scale=uncertainty_scale
    )
    y_hat_uncertainty_95 = get_pred_interval(y_hat_dist)

    num_models = len(y_hat_individual)
    gapfill_columns = {
        'FCH4_F': y_hat_mean,
        'FCH4_F_UNCERTAINTY': y_hat_uncertainty_95[:, 1] - y_hat_mean,
        **{
            f"FCH4_F{i+1}": y_hat_spread[i]
            for i in range(num_models)
        }
    }
    for column, gapfill_data in gapfill_columns.items():
        df[column] = gapfill_data

    return df

def compute_annual_budget(df_gapfilled):
    df_gapfilled.loc[:,'year'] = pd.to_datetime(
        pd.Series(df_gapfilled.index), format='%Y%m%d%H%M').dt.year.values
    
    budget_dict = defaultdict(list)
    for year in df_gapfilled['year'].unique():
        df_year = df_gapfilled[df_gapfilled['year'] == year]
        fch4_columns = [col for col in df_year.columns if 'FCH4_F' in col and 'UNCERTAINTY' not in col]
        annual_budgets = df_year[fch4_columns].sum()
        budget_mean = annual_budgets.mean()
        budget_uncertainty = annual_budgets.std() * 1.96
        budget_dict['year'].append(year)
        budget_dict['budget_mean'].append(budget_mean)
        budget_dict['budget_uncertainty'].append(budget_uncertainty)
    return pd.DataFrame(budget_dict)
