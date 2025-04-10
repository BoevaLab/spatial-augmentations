from pathlib import Path

import rootutils
import scanpy as sc
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def download_zhuang_abca(datasets, abc_cache):
    """
    Ensures all required files for the Zhuang ABCA datasets are downloaded and cached.

    Parameters:
        datasets (list): List of dataset names to process.
        abc_cache (AbcProjectCache): Cache object for accessing and downloading data.
    """
    # download gene metadata
    abc_cache.get_metadata_dataframe(directory=datasets[0], file_name="gene")

    # download cluster annotations
    abc_cache.get_metadata_dataframe(
        directory="WMB-taxonomy", file_name="cluster_to_cluster_annotation_membership_pivoted"
    )
    abc_cache.get_metadata_dataframe(
        directory="WMB-taxonomy", file_name="cluster_to_cluster_annotation_membership_color"
    )

    # download parcellation annotations
    abc_cache.get_metadata_dataframe(
        directory="Allen-CCF-2020",
        file_name="parcellation_to_parcellation_term_membership_acronym",
    )
    abc_cache.get_metadata_dataframe(
        directory="Allen-CCF-2020", file_name="parcellation_to_parcellation_term_membership_color"
    )

    # download dataset-specific files
    for d in datasets:
        print(f"Ensuring files are downloaded for dataset: {d}")

        # download cell metadata
        abc_cache.get_metadata_dataframe(directory=d, file_name="cell_metadata")

        # download CCF coordinates
        abc_cache.get_metadata_dataframe(directory=f"{d}-CCF", file_name="ccf_coordinates")

        # download gene expression matrix
        abc_cache.get_data_path(directory=d, file_name=f"{d}/raw")

    print("All required files have been downloaded and cached.")


def main():
    datasets = ["Zhuang-ABCA-1", "Zhuang-ABCA-2", "Zhuang-ABCA-3", "Zhuang-ABCA-4"]
    download_base = Path("data/domain/abc_atlas/")
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
    download_zhuang_abca(datasets, abc_cache)


if __name__ == "__main__":
    main()
