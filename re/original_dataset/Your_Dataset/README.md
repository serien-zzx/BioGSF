# BioGSF: A Graph-Driven Semantic Feature Integration Framework for Biomedical Relation Extraction
## Additional Requirements
You need to go to [ScispaCy](https://github.com/allenai/scispacy) and download the relevant files, which we use `en_core_sci_lg`.
## Data preparation
You can lay out your data in this folder and make the relevant data by doing the following.
```bash
bash prepare_data.sh
```
Note that your data needs to be in the form of the sample given below:
```
In the [s1] TREM2 [e1] gene, the rs75932628 variant causes C-T base pair changes increases the risk of [s2] Alzheimer's Disease [e2].\tTREM2\tAlzheimer's Disease\tGENE\tDisease\tRelation_type
```