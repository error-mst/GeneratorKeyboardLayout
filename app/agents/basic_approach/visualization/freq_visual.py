from collections import Counter
import matplotlib.pyplot as plt


def create_hist(ax, keys: list[str], values: list[int],
                title: str = "", color: str = "red"):
    ax.bar(keys, values, color=color)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_title(title)