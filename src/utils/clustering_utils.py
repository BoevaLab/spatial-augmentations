import scanpy as sc


def set_leiden_resolution(adata, target_num_clusters, seed, tol=0.01, max_iter=20):
    """
    Adjust the Leiden resolution to match the target number of clusters.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the data.
    target_num_clusters : int
        The desired number of clusters.
    seed : int
        The random seed for reproducibility.
    tol : float, optional
        The tolerance for the difference between the number of clusters and the target. Default is 0.01.
    max_iter : int, optional
        The maximum number of iterations for the binary search. Default is 20.

    Returns:
    -------
    float
        The resolution that matches the target number of clusters.
    """
    low, high = 0.01, 10.0
    best_resolution = None

    for _ in range(max_iter):
        resolution = (low + high) / 2
        sc.tl.leiden(
            adata,
            resolution=resolution,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=seed,
        )
        num_clusters = adata.obs["leiden"].nunique()

        if abs(num_clusters - target_num_clusters) <= tol:
            best_resolution = resolution
            break
        elif num_clusters < target_num_clusters:
            low = resolution
        else:
            high = resolution

    if best_resolution is None:
        best_resolution = (low + high) / 2  # Use the best resolution found within max_iter
        sc.tl.leiden(adata, resolution=best_resolution)

    return best_resolution
