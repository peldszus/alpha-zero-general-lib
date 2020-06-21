import pytest
from alpha_zero_general.utils import parse_game_filename
from alpha_zero_general.utils import parse_model_filename


def test_parse_game_filename():
    assert parse_game_filename("game_123") == 123
    assert parse_game_filename("game_000123") == 123
    assert parse_game_filename("game_000123.file") == 123
    assert parse_game_filename("game_000123_nice") == 123
    assert parse_game_filename("game_000123_nice.file") == 123
    with pytest.raises(ValueError):
        parse_game_filename("game_lala_123")


def test_parse_model_filename():
    assert parse_model_filename("model_123") == 123
    assert parse_model_filename("model_000123") == 123
    assert parse_model_filename("model_000123.file") == 123
    assert parse_model_filename("model_000123_nice") == 123
    assert parse_model_filename("model_000123_nice.file") == 123
    with pytest.raises(ValueError):
        parse_model_filename("model_lala_123")
