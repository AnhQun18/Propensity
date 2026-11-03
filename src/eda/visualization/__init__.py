"""Optional matplotlib plots. Importing eda does not change pyplot backend."""


def axes(ax=None, figsize=(8, 4)):
    if ax is not None:
        return ax.figure, ax
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize)
