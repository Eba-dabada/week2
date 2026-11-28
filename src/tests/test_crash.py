import pandas as pd
import pytest
from src.split import get_split_indices


@pytest.mark.parametrize("bad_y", [
    ["yes", "no", "maybe", "yes", "no", "yes"],
    ["yes", "no", None, "yes", "no", "yes"],
])
def test_get_split_indices_invalid_y_raises(bad_y):
    df = pd.DataFrame({
        "age": range(len(bad_y)),
        "y": bad_y,
    })

    with pytest.raises(ValueError):
        get_split_indices(df)