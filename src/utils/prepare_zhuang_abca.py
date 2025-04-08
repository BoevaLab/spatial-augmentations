import scanpy as sc
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def prepare_zhuang_abca(datasets, abc_cache, output_dir):
    """
    Processes spatial omics data and stores each sample (brain_section_label) in a separate AnnData object.

    Parameters:
        datasets (list): List of dataset names to process.
        abc_cache (AbcProjectCache): Cache object for accessing data.
        output_dir (str): Directory to save the AnnData objects.

    Returns:
        dict: A dict of AnnData objects, one key for each sample.
    """
    # ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load gene metadata
    gene = abc_cache.get_metadata_dataframe(directory=datasets[0], file_name='gene')
    gene.set_index('gene_identifier', inplace=True)

    # load cluster annotations
    cluster_details = abc_cache.get_metadata_dataframe(
        directory='WMB-taxonomy',
        file_name='cluster_to_cluster_annotation_membership_pivoted',
        keep_default_na=False
    ).set_index('cluster_alias')
    cluster_colors = abc_cache.get_metadata_dataframe(
        directory='WMB-taxonomy',
        file_name='cluster_to_cluster_annotation_membership_color'
    ).set_index('cluster_alias')

    # load parcellation annotations
    parcellation_annotation = abc_cache.get_metadata_dataframe(
        directory="Allen-CCF-2020",
        file_name='parcellation_to_parcellation_term_membership_acronym'
    ).set_index('parcellation_index')
    parcellation_annotation.columns = [f'parcellation_{col}' for col in parcellation_annotation.columns]
    parcellation_color = abc_cache.get_metadata_dataframe(
        directory="Allen-CCF-2020",
        file_name='parcellation_to_parcellation_term_membership_color'
    ).set_index('parcellation_index')
    parcellation_color.columns = [f'parcellation_{col}' for col in parcellation_color.columns]

    for d in datasets:
        print(f"Processing dataset: {d}")

        # load cell metadata
        cell_metadata = abc_cache.get_metadata_dataframe(
            directory=d,
            file_name='cell_metadata',
            dtype={"cell_label": str}
        )
        cell_metadata.set_index('cell_label', inplace=True)

        # add cluster annotations and colors
        cell_metadata = cell_metadata.join(cluster_details, on='cluster_alias')
        cell_metadata = cell_metadata.join(cluster_colors, on='cluster_alias')

        # load and add CCF coordinates
        ccf_coordinates = abc_cache.get_metadata_dataframe(
            directory=f"{d}-CCF",
            file_name='ccf_coordinates'
        ).set_index('cell_label')
        ccf_coordinates.rename(columns={'x': 'x_ccf', 'y': 'y_ccf', 'z': 'z_ccf'}, inplace=True)
        cell_metadata = cell_metadata.join(ccf_coordinates, how='inner')

        # add parcellation annotations and colors
        cell_metadata = cell_metadata.join(parcellation_annotation, on='parcellation_index')
        cell_metadata = cell_metadata.join(parcellation_color, on='parcellation_index')

        # load gene expression matrix
        file = abc_cache.get_data_path(directory=d, file_name=f"{d}/raw")
        adata = sc.read_h5ad(file, backed='r')
        gene_expression = adata[:, gene.index].to_df()
        gene_expression.columns = gene.gene_symbol
        gene_expression = gene_expression.reindex(cell_metadata.index)
        adata.file.close()

        # group by brain_section_label (sample) and save each sample as a separate AnnData object
        for sample, sample_metadata in cell_metadata.groupby('brain_section_label'):
            print(f"Processing sample: {sample}")

            # subset gene expression for the sample
            sample_gene_expression = gene_expression.loc[sample_metadata.index]

            # subset spatial coordinates for the sample
            sample_spatial_coords = sample_metadata[['x', 'y']].to_numpy()

            # Create AnnData object for the sample
            sample_adata = sc.AnnData(
                X=sample_gene_expression.values,
                obs=sample_metadata,
            )
            sample_adata.obsm['spatial'] = sample_spatial_coords

            # save the AnnData object to disk with the sample name
            sample_output_path = output_dir / f"{sample}.h5ad"
            sample_adata.write_h5ad(sample_output_path, compression='gzip')
            print(f"AnnData object for sample {sample} saved to {sample_output_path}")


def main():
    datasets = ['Zhuang-ABCA-1', 'Zhuang-ABCA-2', 'Zhuang-ABCA-3', 'Zhuang-ABCA-4']
    download_base = Path('data/domain/abc_atlas/')
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
    output_dir = "data/domain/raw/"
    prepare_zhuang_abca(datasets, abc_cache, output_dir)


if __name__ == "__main__":
    main()