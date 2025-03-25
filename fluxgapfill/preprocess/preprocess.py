import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

from .artificial import learn_gap_dist, sample_artificial_gaps


def preprocess(
        data_dir,
        df,
        split_method='artificial',
        dist='CramerVonMises',
        n_grid=10,
        n_mc=50,
        eval_frac=0.1,
        n_train=10,
        seed=1000
):
    """
    Preprocess the data for model training. This function has no output, and
    writes index lists for the different splits to `data_dir/indices`

    Args:
        data_dir (str): Site-specific directory for all data
        df (pd.DataFrame): Dataframe object containing all relevant site data for
                           modelling.
        split_method (str): How to split the data into training, validation,
                            and test sets.
                            Options: ['artificial', 'random']
        dist (str): Distance measure to use for evaluating the similarity 
                    between the empirical and approximated gap distribution.
                    Options: ['CramerVonMises', 'KolmogorovSmirnoff',
                              'ChiSquare', 'HistogramIntersection',
                              'Hellinger']
        n_grid (int): Width of the grid search per hyperparameter.
                      Must be a positive integer.
        n_mc (int): Number of Monte Carlo interations for estimating
                    the artificial gap distribution.
                    Must be a positive integer.
        eval_frac (float): Proportion of the data to use for testing and validation.
                           Must be a float between 0 and 1.
        n_train (int): Number of paired training and validation sets to
                       generate.
                       Must be a positive integer.
        seed (int): Random seed to initialize pseudorandom number generator.
    """
    data_dir = Path(data_dir)
    print(f"Data preprocessing...")

    df = df.copy().set_index('TIMESTAMP_START')
    df_gap = df[df['FCH4'].isna()]
    df_no_gap = df[df['FCH4'].notna()]

    if split_method == 'artificial':
        artificial_gap_pmf = learn_gap_dist(
            flux_data=df['FCH4'].values,
            dist=dist,
            n_grid=n_grid,
            n_mc=n_mc,
            seed=seed
        )

        # Use the learned distribution to create the
        # training, validation, and test sets

        # Introduce gaps to make the test set
        test_flux = sample_artificial_gaps(
            flux_data=df['FCH4'].values,
            sampling_pmf=artificial_gap_pmf,
            eval_frac=eval_frac,
            seed=seed
        )
        test_indices = np.isnan(test_flux) & df['FCH4'].notna()
        df_test = df[test_indices]
        df_trainval = df[~test_indices]

        train_val_pairs = []
        for i in range(n_train):
            val_flux = sample_artificial_gaps(
                flux_data=df_trainval['FCH4'].values,
                sampling_pmf=artificial_gap_pmf,
                eval_frac=eval_frac,
                seed=seed+i
            )
            val_indices = np.isnan(val_flux) & df_trainval['FCH4'].notna()
            df_val = df_trainval[val_indices]
            df_train = df_trainval[~val_indices & df_trainval['FCH4'].notna()]
            train_val_pairs.append((df_train, df_val))

    elif split_method == 'random':
        df_trainval, df_test = train_test_split(
            df_no_gap, test_size=eval_frac, random_state=seed
        )

        train_val_pairs = []
        for i in range(n_train):
            df_train, df_val = train_test_split(
                df_trainval, test_size=eval_frac, random_state=seed+i
            )
            train_val_pairs.append((df_train, df_val))

    else:
        raise ValueError(f'Splitting method {split_method} not supported.')

    # Assert that the data is evenly split across the sets
    for df_train, df_val in train_val_pairs:
        assert df.shape[0] == (
            df_gap.shape[0] + df_test.shape[0] +
            df_val.shape[0] + df_train.shape[0]
        )

    index_path = data_dir / 'indices'
    os.makedirs(index_path, exist_ok=True)

    np.save(index_path / 'gap.npy', df_gap.index.values)
    np.save(index_path / 'test.npy', df_test.index.values)
    for i, (df_train, df_val) in enumerate(train_val_pairs):
        np.save(index_path / f'train{i}.npy', df_train.index.values)
        np.save(index_path / f'val{i}.npy', df_val.index.values)

    print(f" - Done preprocessing.")
