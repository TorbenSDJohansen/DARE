import random

from typing import Callable, List, Union

from timmsn.data.parsers import SynthParser
from timmsn.data.parsers.parser_sequence_net import setup_formatter


class SynthDateParser(SynthParser):
    def __init__(
            self,
            nb_samples: int,
            label_to_target: Callable,
            year_min: int = 1800,
            year_max: int = 1999,
    ):
        super().__init__(
            nb_samples=nb_samples,
            label_to_target=label_to_target,
            alphabet=None,
            max_seq_len=7,
        )

        self.year_min = year_min
        self.year_max = year_max

    def generate_day(self, month: str) -> str:
        month = int(month)

        if month == 2:
            max_day = 28
        elif month in {4, 6, 9, 11}:
            max_day = 30
        else:
            max_day = 31

        day = str(random.randint(1, max_day))

        if len(day) == 1:
            day = '0' + day

        return day

    def generate_month(self) -> str:
        month = str(random.randint(1, 12))

        return month

    def generate_year(self) -> str:
        year = str(random.randint(self.year_min, self.year_max))

        return year

    def generate_label(self):
        year = self.generate_year()
        month = self.generate_month()
        day = self.generate_day(month)

        date = '-'.join((day, month, year))

        return date


def setup_synthetic_date_parser(
        purpose: str,
        formatter_name: str,
        dataset: str = '',
        num_classes: Union[List[int], None] = None,
        verbose: int = 0,
        formatter_kwargs: Union[List[int], None] = None,
        **kwargs, # pylint: disable=W0613
        ):
    dataset_args = dataset.split('-')[1:]

    try:
        nb_samples = int(dataset_args[-1])
    except ValueError:
        if verbose == 0:
            print('Not specified number of samples per epoch. Using defaults.')

        nb_samples = 500_000 if purpose == 'train' else 1000

    if verbose == 0:
        print(f'Setting up synthetic dataset {dataset}. Samples per epoch: {nb_samples}')

    transform_label, clean_pred, _num_classes = setup_formatter(
        formatter_name=formatter_name,
        formatter_kwargs=formatter_kwargs,
        verbose=verbose,
        )

    if purpose not in {'train', 'test', 'predict'}:
        raise ValueError('purpose must be train, test, or predict, got {purpose}')

    if num_classes is None:
        num_classes = _num_classes
        if verbose == 0:
            print(f'`--num-classes` not specified. Using formatter default: {_num_classes}')

    parser = SynthDateParser(
        nb_samples=nb_samples,
        label_to_target=transform_label,
        )

    if purpose == 'train':
        parser_eval = SynthDateParser(
            nb_samples=max(1000, min(50_000, int(0.1 * len(parser)))), # in interval [1000, 50000]
            label_to_target=transform_label,
            )
    else:
        parser_eval = None

    return parser, parser_eval, clean_pred, num_classes
