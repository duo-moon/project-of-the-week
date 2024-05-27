import argparse
import pathlib

from sklearn.ensemble import RandomForestRegressor

from helpers import load_yaml, dump_pickle, load_pickle


def run_train(feature_path: pathlib.Path, params_path: pathlib.Path, output_path: pathlib.Path) -> None:
    X_train, y_train = load_pickle(feature_path.joinpath('train.pkl'))
    best_params = load_pickle(params_path.joinpath('best_params.pkl'))

    best_params = {key: int(value) for key, value in best_params.items()}

    rf = RandomForestRegressor(**best_params)
    rf.fit(X_train, y_train)

    output_path.mkdir(exist_ok=True)
    dump_pickle(rf, output_path.joinpath('model.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)['train']
    run_train(
        feature_path=pathlib.Path(config['features_path']),
        params_path=pathlib.Path(config['params_path']),
        output_path=pathlib.Path(config['output_path']),
    )
