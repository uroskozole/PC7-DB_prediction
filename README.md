# README #

# RelAtional Learning On Graphs (REALOG) PACKAGE INSTALLATION

```bash
pip install -e .
```

# REALOG PACKAGE USAGE

After installing the package, you can use the `/src/train.py` to train GNN models on your data.

First, upload your data in the `/data/<dataset-name>` folder. The data should be in the form of a `.csv` file for each table. In the same folder there has to be a `metadata.json` file, following the SDV multi-table format, that describes the relationships between the tables. The SDV metadata specification can be found [here](https://docs.sdv.dev/sdv/reference/metadata-spec/multi-table-metadata-json).

At the end of the `train.py` file, you should modify the following variables to fit your data:

```python
    dataset = '<dataset-name>'
    target_table = '<table-name>'
    target = '<target-column-name>'
    task = '<predictive-task-type>' # 'classification' or 'regression'
    model_name = '<model-name>' # GIN, GAT, GATv2, GraphSAGE, EdgeCNN
```
# STRUCTURE OF THE REPO

- `data/`: Folder containing the data and metadata files.
- `final_report`: Folder containing the final report.
- `src/`: Folder containing the source code.
- `src/realog/`: Folder containing the source code of the REALOG package.
- `src/baselines/`: Folder containing the scripts to run the baselines.
- `src/HPC/`: Folder containing the scripts to run the experiments on the HPC.
- `src/scripts/`: Folder containing the scripts to split the used datasets into train/val/test.
- `src/train.py`: Script to train the GNN models.