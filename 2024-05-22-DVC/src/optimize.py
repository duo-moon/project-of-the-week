import argparse
import pathlib

import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from helpers import load_pickle, dump_pickle, load_yaml


def run_optimization(features_path: pathlib.Path, output_path: pathlib.Path, n_trials: int, seed: int) -> None:
    X_train, y_train = load_pickle(path=features_path.joinpath('train.pkl'))
    X_val, y_val = load_pickle(path=features_path.joinpath('val.pkl'))

    def objective(params: dict) -> dict:
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.uniform('max_depth', 1, 20)),
        'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 1, 5)),
        'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 10)),
        'n_estimators': scope.int(hp.uniform('n_estimators', 100, 5000)),
        'random_state': seed,
    }

    rstate = np.random.default_rng(seed)

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=Trials(),
        rstate=rstate,
    )

    output_path.mkdir(exist_ok=True)
    dump_pickle(best, output_path.joinpath('best_params.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)['optimize']

    run_optimization(
        features_path=pathlib.Path(config['features_path']),
        output_path=pathlib.Path(config['output_path']),
        n_trials=config['n_trials'],
        seed=config['seed'],
    )
