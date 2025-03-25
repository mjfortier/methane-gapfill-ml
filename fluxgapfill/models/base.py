import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

from fluxgapfill.metrics import metric_dict


def add_wind_predictors(df):
    df = df.copy()
    df['WD'] = df['WD'] / 180 * np.pi
    notna = df['WD'].notna()
    df.loc[notna, 'WD_sin'] = np.sin(df.loc[notna, 'WD'])
    df.loc[notna, 'WD_cos'] = np.cos(df.loc[notna, 'WD'])
    df.drop('WD', axis=1, inplace=True)
    return df


def add_temporal_predictors(df):
    # Assumes that the index is Ameriflux datetime, ex: 201906120330
    df = df.copy()
    timestamp = pd.to_datetime(pd.Series(df.index), format='%Y%m%d%H%M')
    doy = timestamp.dt.dayofyear
    tod = timestamp.dt.hour
    tod += 0.5 * (timestamp.dt.minute.astype(float) == 30).astype(float)

    doy_sin = np.sin(2 * np.pi * (doy - 1) / 366)
    doy_cos = np.cos(2 * np.pi * (doy - 1) / 366).rename('doy_cos')
    tod_sin = np.sin(2 * np.pi * (tod - 1) / 24).rename('tod_sin')
    tod_cos = np.cos(2 * np.pi * (tod - 1) / 24).rename('tod_cos')

    df.loc[:,'doy_sin'] = doy_sin.values
    df.loc[:,'doy_cos'] = doy_cos.values
    df.loc[:,'tod_sin'] = tod_sin.values
    df.loc[:,'tod_cos'] = tod_cos.values
    return df


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, predictors, cv=5, n_iter=20):
        self.predictors = predictors
        self.cv = cv
        self.n_iter = n_iter
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def preprocess(self, df):
        """Prepare df to be input to the model."""
        df = df[self.predictors]
        df = add_temporal_predictors(df)

        if 'WD' in self.predictors:
            df = add_wind_predictors(df)
        return df

    def fit(self, df, target):
        """Train on a training set and select optimal hyperparameters."""
        y = df[target]
        df = self.preprocess(df)

        df.loc[:, :] = self.imputer.fit_transform(df)
        if self.scaler is not None:
            df.loc[:, :] = self.scaler.fit_transform(df)
        
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        
        random_search.fit(df, y)
        self.model = random_search.best_estimator_

    def predict(self, df):
        df = self.preprocess(df)
        df.loc[:, :] = self.imputer.transform(df)
        if self.scaler is not None:
            df.loc[:, :] = self.scaler.fit_transform(df)
        
        return self.model.predict(df)

    def evaluate(self, df, target, metrics):
        y = df[target]
        y_hat = self.predict(df)
        scores = []
        for metric in metrics:
            if metric not in metric_dict:
                raise ValueError(f"Metric {metric} not supported.")
            metric_fn = metric_dict[metric]
            scores.append(metric_fn(y, y_hat))
        return scores

    def save(self, path):
        """Save model to path."""
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    @property
    def feature_importances(self):
        weights = self.model.feature_importances_
        return dict(sorted(zip(
            self.predictors, weights),
            key=lambda x: x[-1],
            reverse=True
        ))
