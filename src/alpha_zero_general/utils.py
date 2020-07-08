"""
Utility methods
"""

import re


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


MODEL_FILENAME_PATTERN = re.compile(r"model_(\d+)(_.*)?(\..*)?")
GAME_FILENAME_PATTERN = re.compile(r"game_(\d+)(_.*)?(\..*)?")


def parse_model_filename(filename):
    """Returns the model revision for a model filename like 'model_00001'
    or 'model_1234.file' or 'model_0123_keep'."""
    match = re.match(MODEL_FILENAME_PATTERN, filename)
    if match:
        return int(match.groups()[0])
    raise ValueError(
        f"Cannot parse the model revision from model filename '{filename}''."
    )


def parse_game_filename(filename):
    """Returns the running game number for a game filename like 'game_00001'
    or 'game_1234.file' or 'game_0123_keep'."""
    match = re.match(GAME_FILENAME_PATTERN, filename)
    if match:
        return int(match.groups()[0])
    raise ValueError(
        f"Cannot parse the game number from game filename '{filename}'."
    )
