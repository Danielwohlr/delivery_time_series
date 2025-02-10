# food_delivery_ts

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


Presentation is in reports/presentation.pdf.

Installation instructions: 
Being in the root of this project, run
```conda env create -f environment.yml``` in your terminal to create a conda environment.

Then activate it and run, e.g. src/modeling/evaluate.py or src/modeling/qual_analysis.py, for running the experiments.



## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data                <- The original, immutable data dump.
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         food_delivery_ts and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                <- Source code for use in this project.
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── features.py             <- Code to create features for modeling
    └── plots.py                <- Code to create some visualizations
    ├── modeling                
    │   ├── __init__.py 
    │   ├── evaluate.py          <- Compute CrossValidation Losses
    │   └── qual_analysis.py            <- Qualitative Analysis, e.g. Analysis predicted values, Residual analysis, Feature Importance
    │
    ├── models                  <- Modules with all Prediction models definitions                
        ├── ewm.py              <- benchmark ewm
        ├── hgbr.py             <- Gradient Boosting Regression Tree
        ├── regression.py        <- All Implementations of Linear Regression Pipelines
```

--------

