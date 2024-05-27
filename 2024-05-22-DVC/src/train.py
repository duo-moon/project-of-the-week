import argparse
import pathlib

from sklearn.ensemble import RandomForestRegressor

from helpers import load_yaml, dump_pickle, load_pickle


def run_train(feature_path: pathlib.Path, model_path: pathlib.Path, n_estimators: int, seed: int) -> None:
    X_train, y_train = load_pickle(feature_path.joinpath('train.pkl'))

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train, y_train)

    model_path.mkdir(exist_ok=True)
    dump_pickle(rf, model_path.joinpath('model.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)['train']
    run_train(
        feature_path=pathlib.Path(config['features_path']),
        model_path=pathlib.Path(config['model_path']),
        n_estimators=config['n_estimators'],
        seed=config['seed'],
    )
