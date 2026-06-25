# ml-pagerankfas

ml-pagerankfas/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── pagerankfas/
│       ├── __init__.py
│       ├── classic.py
│       ├── ml_enhanced/
│       │   ├── __init__.py
│       │   ├── data_generator.py
│       │   ├── features.py
│       │   ├── dataset.py
│       │   ├── model.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── batch_pagerank.py
│       └── utils/
│           ├── __init__.py
│           ├── scc_handler.py
│           └── line_graph.py
├── scripts/
│   ├── generate_data.py
│   └── run_training.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
└── tests/
    ├── __init__.py
    ├── test_classic_pagerankfas.py
    └── datasets/
        └── cit-HepPh.txt.gz