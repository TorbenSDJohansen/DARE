"""
@author: sa-tsdj

Given workspace from ENS TypeApp, convert to format usable as label.

"""


import argparse
import os

from typing import Tuple, Union

import json

from sklearn.model_selection import train_test_split

import pandas as pd

from grade_formatter import ALLOWED_GRADE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--workspaces', type=str, nargs='+', default=None)
    parser.add_argument('--wsp-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--share-test', type=float, default=None)
    parser.add_argument('--nb-test', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--add-filename-hash-to-seed', action='store_true', default=False,
                        help='fixes issue of exact same split across all workspaces of same lenght while still preserving reproducibility')

    args = parser.parse_args()

    if not (args.workspaces is None) ^ (args.wsp_dir is None):
        raise ValueError('Must specify exactly one of --workspaces or -wsp-dir!')

    if args.wsp_dir is not None and not os.path.isdir(args.wsp_dir):
        raise NotADirectoryError(f'--wsp-dir {args.wsp_dir} does not exist')

    if args.wsp_dir is None:
        for file in args.workspaces:
            if not os.path.isfile(file):
                raise FileNotFoundError(f'requested workspace {file} does not exist')

    if args.wsp_dir is not None:
        workspaces = [os.path.join(args.wsp_dir, x) for x in os.listdir(args.wsp_dir) if x.endswith('.json')]
        args.workspaces = workspaces

    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        raise NotADirectoryError(f'--output-dir {args.output_dir} does not exist')

    if not (args.share_test is None) ^ (args.nb_test is None):
        raise ValueError('Must specify exactly one of --share-test or -nb-test!')

    if args.share_test is not None and not 0 <= args.share_test <= 1:
        raise ValueError(f'--share-test must be in [0, 1], got {args.share_test}')

    if args.nb_test is not None and args.nb_test < 0:
        raise ValueError(f'--nb-test must be 0 or positive, got {args.nb_test}')

    if args.seed < 0:
        raise ValueError(f'--seed must be 0 or positive, got {args.seed}')

    return args


def load_workspace(file: str) -> pd.DataFrame:
    with open(file, 'r', encoding='utf-8') as stream:
        workspace = json.load(stream)

    cursor = workspace['cursor'] + 1 # image reached in workspace
    workspace = workspace['elements']
    workspace = workspace[:cursor] # only keep images reached

    labels = pd.DataFrame({
        'image_id':[x['name'] for x in workspace], # '0024.jpg'
        'file': [x['path'] for x in workspace], # 'path/to/0024.jpg'
        'label': [x['properties'].get('default_tag', '') for x in workspace], # '123'
        })
    # NOTE: Some fields left empty miss default_tag, but not sure why, above
    # .get() is fix: Note this is *not* the case for all fields left empty,
    # only for some, which is what is puzzling.

    # Empty values currently coded as empty spring, instead use
    # more explicit name
    labels['label'] = labels['label'].replace('', 'empty')

    # Dominated by empty values - let us make version without
    labels = labels[labels['label'] != 'empty']
 
    # Drop values not in allowed list
    labels = labels[labels['label'].isin(ALLOWED_GRADE)]

    return labels


def split(
        labels: pd.DataFrame,
        test_size: Union[int, float],
        seed: int,
        ) -> Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    if test_size == 0:
        return labels, None

    if isinstance(test_size, float) and test_size == 1:
        return None, labels

    if test_size > len(labels):
        raise ValueError(f'specified test size {test_size} is larger than number of labels {len(labels)}')

    return train_test_split(labels, test_size=test_size, random_state=seed)


def workspace_to_labels(
        file: str,
        output_dir: str,
        test_size: Union[int, float],
        seed: int,
        add_filename_hash_to_seed: bool,
        ):
    labels = load_workspace(file)
    labels = labels[['image_id', 'label']]

    basename = os.path.basename(file)

    if add_filename_hash_to_seed:
        add = hash(basename)
        add = add % (2 ** 32 - 1 - seed) # ensures seed in [0, 2 ** 32 - 1]
        seed += add

    train, test = split(labels=labels, test_size=test_size, seed=seed)

    if train is not None:
        train.to_csv(
            os.path.join(output_dir, 'train', basename.replace('.json', '.csv')),
            index=False,
            )
    if test is not None:
        test.to_csv(
            os.path.join(output_dir, 'test', basename.replace('.json', '.csv')),
            index=False,
            )


def main():
    args = parse_args()
    test_size = args.share_test or args.nb_test

    for file in args.workspaces:
        workspace_to_labels(
            file=file,
            output_dir=args.output_dir,
            test_size=test_size,
            seed=args.seed,
            add_filename_hash_to_seed=args.add_filename_hash_to_seed,
            )


if __name__ == '__main__':
    main()
