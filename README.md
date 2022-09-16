### spliceJAC: Identify transition driver genes and cell state specific regulatory interactions from single-cell transcriptome data

spliceJAC is a python-based toolkit to reconstruct cell state-specific gene regulatory networks (GRN) and predict transition driver genes leading to cell differentiation.

### Overview of spliceJAC

spliceJAC requires the spliced (S) and unspliced (U) count matrices from single cell RNA-sequencing (scRNA-seq) experiment as well as cell annotations that are used to identify the cell states in the dataset. Starting from this information, spliceJAC builds a local mRNA splicing model that provides information about cell state specific gene regulatory interactions and driver genes that induce cell state transitions.

![spliceJAC schematic](misc/spliceJAC.png)

### Applications

Application of spliceJAC include:

- Investigate the context-specific signaling role of genes in different cell states.

- Analyze multi-stable systems where several cell states can coexist.

- Distinguish the transition driver genes leading to distinct differentiation paths stemming from a common initial state.

### Use and installation

#### Install the package from PyPl

spliceJAC is available as a python library and can be installed with pip:

```console
pip install -U splicejac
```

#### Working within the spliceJAC repository

In the meantime, you can zip download the spliceJAC repository from the green 'Code' bottom at the top right corner of this page and run your custom code and notebooks from within the repository.

#### Setting up a virtual environment 

Whether you decide to install the package or to work within the spliceJAC Repo, we suggest to work within a virtual environment to avoid conflict and ensure that all dependencies are updated to the required version. Guidelines to create a virtual environment in Python can be found [here](https://docs.python.org/3/library/venv.html).

spliceJAC requires the installation of dependencies including [Numpy](https://numpy.org), [Matplotlib](https://matplotlib.org), [Pandas](https://pandas.pydata.org), [Scanpy](https://pandas.pydata.org) and [scVelo](https://scvelo.readthedocs.io). The full list of required packages can be found in the requirements.txt file within this folder. Once your virtual environment is set up, you can install all required dependencies by running:

```console
pip install -r requirements.txt
```

### Documentation

You can check the spliceJAC documentation [here](https://splicejac.readthedocs.io/en/latest/). This documentation includes two in-depth notebook tutorials to demonstrate spliceJAC's applications in inferring cell state specific gene regulatory networks and analyze cell state transitions:

- [GRN inference](https://splicejac.readthedocs.io/en/latest/notebooks/GRN%20Inference.html)
- [Transitions](https://splicejac.readthedocs.io/en/latest/notebooks/Transitions.html)

Note: the documentation is currently under heavy development, check frequently for updates!

### References

Federico Bocci, Peijie Zhou, Qing Nie, spliceJAC: Identify transition driver genes and cell state specific regulatory interactions from single-cell transcriptome data, Preprint (2022).

The detailed testing and benchmarking of spliceJAC is publicly available [here](https://github.com/cliffzhou92/jacobian-inference-benchmarking).

### Further reading

Bergen et al. (2020), Generalizing RNA velocity to transient cell states through dynamical modeling, [Nature Biotechnology](https://www.nature.com/articles/s41587-020-0591-3).

Pratapa et al. (2020), Benchmarking algorithms for gene regulatory network inference from single cell transcriptomic data, [Nature Methods](https://www.nature.com/articles/s41592-019-0690-6).

Wolf et al. (2018), SCANPY: large-scale single-cell gene expression data analysis, [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0).

### Contact and support

Questions or suggestions are welcomed and highly appreciated. You can contact us by opening an issue or via email at fbocci@uci.edu.
