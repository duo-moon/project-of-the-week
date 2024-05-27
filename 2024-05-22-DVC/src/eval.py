import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from dvclive import Live
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from helpers import load_yaml, load_pickle


def save_importance_plot(live: Live, model: RandomForestRegressor, feature_names: list[str]):
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)


def evaluate(model_path: pathlib.Path, features_path: pathlib.Path, live: Live):
    model = load_pickle(path=model_path)
    X_val, y_val = load_pickle(features_path.joinpath('val.pkl'))

    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    live.log_metric('rmse', rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)['eval']

    dv = load_pickle(path=pathlib.Path(config['features_path']).joinpath('vectorizer.pkl'))
    model = load_pickle(path=pathlib.Path(config['model_path']))

    with Live(config['eval_output_path']) as live:
        evaluate(
            model_path=pathlib.Path(config['model_path']),
            features_path=pathlib.Path(config['features_path']),
            live=live,
        )
        save_importance_plot(
            live=live,
            model=model,
            feature_names=dv.feature_names_,
        )
