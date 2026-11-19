"""Fast offline environment tests."""

import pytest


def test_numpy():
    import numpy as np

    assert np.array([1, 2, 3]).sum() == 6


def test_astropy_units():
    import astropy.units as u

    assert (1000 * u.m).to(u.km).value == 1


def test_torch():
    # PyTorch is installed separately; skip this test when it is absent.
    torch = pytest.importorskip("torch", reason="PyTorch is not installed")
    assert torch.tensor([1, 2, 3]).sum().item() == 6


def test_sqlalchemy_sqlite():
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE test (id INTEGER)"))
    engine.dispose()


@pytest.mark.parametrize(
    "module",
    ["scipy", "pandas", "matplotlib", "astroquery", "lightkurve", "pyvo", "h5py", "pyarrow"],
)
def test_scientific_stack_imports(module):
    __import__(module)


def test_project_imports():
    import src.data_collection.pipeline  # noqa: F401
    import src.database.repository  # noqa: F401
    import src.utils.fits_utils  # noqa: F401
