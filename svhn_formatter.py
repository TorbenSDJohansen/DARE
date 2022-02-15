# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import numpy as np

from timmsn.data.formatters import register_formatter, NumSeqFormatter


def _sanitize(raw_input: str or int or float):
    if raw_input is None:
        return None

    if isinstance(raw_input, float):
        assert int(raw_input) == raw_input, raw_input # assert no decimals

    raw_input = int(raw_input) # this also removes leading 0s from str
    raw_input = str(raw_input)

    return raw_input


class NumSeqToDateLikeFormatter: # pylint: disable=C0115
    def __init__(self):
        self.max_len = 5
        self.min_len = 1
        self.num_classes = [4, 10, 14, 11, 11, 11, 11] # To match DDMYY model
        self.map_digit_idx_to_pos = {0: 1, 1: 3, 2: 4, 3: 5, 4: 6} # Not use first day and month "tokens"

    def transform_label(self, raw_input: int or float or str) -> np.ndarray: # pylint: disable=C0116
        '''
        Map number sequence to "date-like" format, left to right, not using
        the first day and the month "tokens".

        Mapping examples:
            (1) "123" -> [0, 1, 0, 2, 3, 10, 10]
            (2) "4" -> [0, 4, 0, 10, 10, 10, 10]
            (3) "12345" -> [0, 1, 0, 2, 3, 4, 5]

        In general, output of form [0, X, 0, X, X, X, X]
        '''
        mod_input = _sanitize(raw_input)

        if mod_input is None:
            return mod_input

        assert self.max_len >= len(mod_input) >= self.min_len
        label = [0, 0, 0, 10, 10, 10, 10]

        for i, digit in enumerate(mod_input):
            label[self.map_digit_idx_to_pos[i]] = int(digit)

        # Assert consistency.
        assert mod_input == str(self.clean_pred(label, False)), self.clean_pred(label, False)

        label = np.array(label).astype('float')

        return label

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> int: # pylint: disable=C0116
        # Extract only relevant part
        pred = [raw_pred[i] for i in self.map_digit_idx_to_pos.values()]

        # Delete 10s (i.e. no number). Note first number is never 10.
        pred = [x for x in pred if x != 10]

        # NOTE: First token should never be zero, but technically possible now.
        # Maybe cast 0 -> 1? First digit probably most likely 1.

        clean = []

        for val in pred:
            clean.append(str(val))

        clean = int(''.join(clean))

        # Need to be cycle consistent - however, the function may be called from
        # `transform_label`, and we do not want infinite recursion, hence the if.
        if assert_consistency:
            transformed_clean = self.transform_label(clean)
            if not (transformed_clean is None or all(raw_pred.astype('float') == transformed_clean)):
                raise Exception(raw_pred, pred, clean, transformed_clean)

        return clean


@register_formatter
def svhn_as_date() -> NumSeqToDateLikeFormatter:
    return NumSeqToDateLikeFormatter()


@register_formatter
def svhn_as_numseq() -> NumSeqFormatter:
    return NumSeqFormatter(5, 1)


def _test_formatter():
    formatters = (svhn_as_numseq(), svhn_as_date())

    for formatter in formatters:
        for i in range(1, 100_000):
            for cast_to_type in (int, float, str):
                label = formatter.transform_label(cast_to_type(i))
                original = formatter.clean_pred(label.astype(int))
                assert int(i) == original


if __name__ == '__main__':
    _test_formatter()
