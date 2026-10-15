import matplotlib
import matplotlib.pyplot as plt
import pytest

# Plots in tests must never open a window
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
