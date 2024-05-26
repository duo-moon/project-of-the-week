import argparse
import pathlib
import pickle

import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


def load_yaml(path: pathlib.Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_pickle(obj, path: pathlib.Path) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def preprocess_and_save_data(data_path: pathlib.Path, output_path: pathlib.Path, seed: int, split_size: float) -> None:
    data = pd.read_csv(data_path)
    x = data[['experience_level', 'employment_type', 'job_title', 'company_size']]
    y = data['salary_in_usd']

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=split_size, random_state=seed)

    dv = DictVectorizer()
    X_train = dv.fit_transform(X_train.to_dict(orient='records'))
    X_val = dv.transform(X_val.to_dict(orient='records'))

    output_path.mkdir(exist_ok=True)
    dump_pickle(dv, output_path.joinpath('vectorizer.pkl'))
    dump_pickle((X_train, y_train.values), output_path.joinpath('train.pkl'))
    dump_pickle((X_val, y_val.values), output_path.joinpath('val.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)['process']
    preprocess_and_save_data(
        data_path=pathlib.Path(config['data_path']),
        output_path=pathlib.Path(config['output_path']),
        seed=config['seed'],
        split_size=config['test_split_size'],
    )
