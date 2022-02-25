# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import numpy as np

from timmsn.data.formatters import register_formatter


def _sanitize(raw_input: str):
    assert isinstance(raw_input, str), raw_input

    raw_input = raw_input.replace('-', '')

    return raw_input


class NumSeqFormatter: # pylint: disable=C0115
    def __init__(self):
        self.max_len = 5
        self.min_len = 1
        self.num_classes = [10, 11, 11, 11, 11]

    def transform_label(self, raw_input: str) -> np.ndarray: # pylint: disable=C0116
        mod_input = _sanitize(raw_input)

        if len(mod_input) > self.max_len: # Restrict to max. len. of 5
            print(mod_input)
            return None

        assert self.max_len >= len(mod_input) >= self.min_len, mod_input

        label = []

        for digit in mod_input:
            label.append(int(digit))

        label += [10] * (self.max_len - len(mod_input))

        # Assert consistency.
        assert raw_input == self.clean_pred(label, False)

        label = np.array(label).astype('float')

        return label

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> int: # pylint: disable=C0116
        # Delete 10s (i.e. no number). Note first number is never 10.
        pred = raw_pred.copy()
        pred = [x for x in pred if x != 10]

        raw_pred_mod = np.array(pred + [10] * (self.max_len - len(pred)))

        clean = []

        for val in pred:
            clean.append(str(val))

        clean = str('-'.join(clean))

        # Need to be cycle consistent - however, the function may be called from
        # `transform_label`, and we do not want infinite recursion, hence the if.
        if assert_consistency:
            transformed_clean = self.transform_label(clean)
            if not (transformed_clean is None or all(raw_pred_mod.astype('float') == transformed_clean)):
                raise Exception(raw_pred, pred, clean, transformed_clean)

        return clean


@register_formatter
def svhn_as_numseq() -> NumSeqFormatter:
    return NumSeqFormatter()


def _test():
    formatter = svhn_as_numseq()

    for raw in ('1-2-3', '1-2', '0-0-1', '0', '1-2-3-4-5-6'):
        label = formatter.transform_label(raw)
        clean = formatter.clean_pred(label.astype(int))
        assert raw == clean


if __name__ == '__main__':
    _test()
