import torch
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.colors as mcolors


def kmeans_grad(
    img: torch.Tensor,
    num_clusters: int,
    num_iter: int,
    learning_rate: float,
) -> torch.Tensor:

    # Flatten img tensor for easier processing
    # TODO: use img_flat = img.view(3, -1).T
    img_flat = img.flatten(start_dim=1).T

    # Forgy initialization; Pick distinct samples as initialization for the
    # centroids.
    centroids = img_flat[torch.randperm(len(img_flat))[:num_clusters]].requires_grad_()

    for i in range(num_iter):
        distances = torch.cdist(img_flat, centroids)
        assert tuple(distances.shape) == (img_flat.shape[0], num_clusters)

        centroid_idx = distances.argmin(dim=1)
        loss = torch.sum(distances.gather(1, centroid_idx.unsqueeze(dim=1)).squeeze())
        loss.backward()

        with torch.no_grad():
            centroids -= learning_rate * centroids.grad
        centroids.grad.zero_()

        print("\r"*100 + f"it {i}, loss {loss.item()}", end="")

    return centroids.detach()


def plot_spectrum(rgb: torch.Tensor) -> ggplot:
    assert len(rgb.shape) == 2 \
        and rgb.shape[1] == 3 \
        and torch.all((0.0 <= rgb) & (rgb <= 1.0))


    brightness = torch.norm(rgb, dim=1)
    rgb = rgb[brightness.argsort()]
    colors = [mcolors.rgb2hex(rgb.tolist()) for rgb in rgb]

    # Create a DataFrame where each row represents a color and its position in the spectrum
    df = pd.DataFrame({'color': colors, 'x': range(len(colors)), 'y': [1]*len(colors)})

    # Create a color spectrum with vertical bars
    plot = (ggplot(df, aes('x', 'y', fill='color'))
            + geom_tile(color="none")
            + scale_fill_identity()
            + theme_void()
            + theme(legend_position='none'))

    return plot


def kmeans_encode(
    img: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Returns the centroids' indices of its pixels."""

    assert img.shape[0] == 3 and len(img.shape) == 3

    img_flat = img.view(3, -1).T
    distances = torch.cdist(img_flat, centroids)
    centroid_idx = distances.argmin(dim=1)
    assert tuple(centroid_idx.shape) == (img.shape[1] * img.shape[2],)

    return centroid_idx.reshape((img.shape[1], img.shape[2]))

def kmeans_decode(
    encoded_img: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:

    result = centroids[encoded_img].permute(2, 0, 1)
    return result
