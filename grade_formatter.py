"""
@author: sa-tsdj

"""


import numpy as np

from timmsn.data.formatters.registry import register_formatter


__all__ = ['GradeFormatter']

ALLOWED_GRADE = {
    'a', 'ab', 'ab+', 'b', 'b-', 'b+', 'ba', 'ba-', 'ba+', 'bc',
    # 'ab-',
    'other', 'empty', 'ditto',
}
MAP_GRADE_TOKEN = {x: i for i, x in enumerate(sorted(ALLOWED_GRADE))}
MAP_TOKEN_GRADE = {v: k for k, v in MAP_GRADE_TOKEN.items()}


def _sanitize(raw_input: str):
    if raw_input is None:
        return None

    if not isinstance(raw_input, str):
        raise TypeError(
            f'Raw input must be string or None, received input {raw_input}' +
            f' of type {type(raw_input)}.'
            )

    if raw_input == 'bad': # Segmentation gone wrong, drop
        return None

    assert raw_input in ALLOWED_GRADE, raw_input

    return raw_input


def transform_label(raw_input: str) -> np.ndarray: # pylint: disable=C0116
    grade = _sanitize(raw_input)

    if grade is None:
        return None

    label = MAP_GRADE_TOKEN[grade]
    label = np.array([label])

    cleaned = clean_pred(label, False)

    if grade != cleaned:
        raise ValueError(raw_input, grade, label, cleaned)

    return label.astype('float')


def clean_pred(raw_pred: np.ndarray, assert_consistency: bool = True) -> str: # pylint: disable=C0116
    assert len(raw_pred) == 1

    clean = MAP_TOKEN_GRADE[raw_pred[0]]

    if assert_consistency:
        transformed_clean = transform_label(clean)

        if not (transformed_clean is None or all(raw_pred.astype('float') == transformed_clean)):
            raise ValueError(raw_pred, clean, transformed_clean)

    return clean


class GradeFormatter(): # pylint: disable=C0115, R0903
    def __init__(self):
        self.transform_label, self.clean_pred = transform_label, clean_pred
        self.num_classes = [max(MAP_TOKEN_GRADE) + 1]


@register_formatter
def grades() -> GradeFormatter:
    return GradeFormatter()
