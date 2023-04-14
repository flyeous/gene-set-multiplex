## Dive into gene-set multiplex networks facilitated by a clustering-based measure of similarity
To reproduce the experiments.

## :cactus: Where to find the dataset

* Download processed scRNA-seq data ([Domínguez Conde, C., et al.](https://www.tissueimmunecellatlas.org))

* Download GO (C5) and ImmuneSig (C7) gene sets (gene symbols) from [Molecular Signatures Database](https://www.gsea-msigdb.org/gsea/msigdb/index.jsp).


## :clipboard: Code environment

`conda env create -f gs.yml`

Install R-4.2.1 or newer release, and then run

`pip install rpy2`

Requirements for installing and using [grimon](https://github.com/mkanai/grimon) package. 


## :coffee: Demos

* **Instantiate a series of similarity matrices** ([code](https://github.com/flyeous/gene-set-multiplex/blob/main/similarity_matrix.ipynb))
* **Procedures in the multiplex network analysis** ([code](https://github.com/flyeous/gene-set-multiplex/blob/main/multiplex_network_analysis.ipynb))
* **Snapshots of the interations between two groups of gene sets**([code](https://github.com/flyeous/gene-set-multiplex/blob/main/coexpression_GO_CC_Immunesig.ipynb))
* **Running time comparison and performace evaluations** ([code](https://github.com/flyeous/gene-set-multiplex/blob/main/Program_run_duration_comparison.ipynb))
* **Applications in gene set enrichment analysis** ([code](https://github.com/flyeous/gene-set-multiplex/blob/main/Application_in_GSEA.ipynb))

## :e-mail: Contact 
* zheng.cheng.68e@st.kyoto-u.ac.jp