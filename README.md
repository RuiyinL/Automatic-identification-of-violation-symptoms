# Replication Package for the Paper: Towards Automatic Identification of Violation Symptoms of Architecture Erosion: Experiments and Industrial Validation

##### Authors: Ruiyin Li, Peng Liang, Paris Avgeriou

## Abstract of this Study
During code review, reviewers typically expend a substantial amount of effort in comprehending code changes, since significant information (e.g., architecture violations) for inspecting code changes may be dispersed across several files that the reviewers are not acquainted with. Automatic identification of architecture violations from the discussions of code reviews could be necessary and time-saving for developers to locate and check the potential architecture violations. In this paper, we developed 15 machine learning-based and 4 deep learning-based classifiers with three pre-trained word embeddings to identify violation symptoms of architecture erosion from developer discussions in code reviews (i.e., code review comments from four large open-source projects from the OpenStack (Nova and Neutron) and Qt (Qt Base and Qt Creator) communities). We then conducted a survey that acquired the feedback from the involved participants who discussed architecture violations in code reviews to validate the usefulness of our trained ML-based and DL-based classifiers. The results show that the SVM classifier based on _word2vec_ pre-trained word embedding performs the best with an F1-score of 0.779. In most cases, classifiers with the _fastText_ pre-trained word embedding model can achieve relatively good performance. Furthermore, 200-dimensional pre-trained word embedding models outperform classifiers that use 100 and 300-dimensional models. In addition, an ensemble classifier based on the majority voting strategy can further enhance the classifier and outperforms the individual classifiers. Finally, an online survey of the involved developers reveals that the violation symptoms identified by our approaches have practical meanings and can provide warnings for developers.

## Structure of the Replication Package

（TBA）

```
├── LICENSE
├── README.md
├── data
│   ├── extracted_features
│   |   ├── FastText_100_non_violation.csv
│   |   ├── FastText_100_violation.csv
│   |   ├── ...
│   ├── word_embedding
│   |   ├── cc.en.100.bin
│   |   ├── cc.en.200.bin
│   |   ├── cc.en.300.bin
│   |   ├── glove.twitter.27B.200d.txt
│   |   ├── SO_vectors_200.bin
│   |   ├── embedding_dim.py
│   |   └── Download_url.txt
│   ├── Randomly_selected_comments.xlsx
│   ├── Violation symptoms.xlsx
└── scripts
    ├── requirements.txt
    ├── classifiers
    |   ├── DL_classifiers.py
    |   ├── DL_models.py
    |   ├── DL_utility.py
    |   └── ML_classifiers.py
    └── preprocessing
        ├── feature_extraction.py
        ├── managedb.py
        ├── preprocessing.py
        └── w2vemb.py
```

## Experiment Steps

(TBA)

<!-- 
Step 1: Preprocessing and feature extraction.

- Run `feature_extraction.py` to conduct preprocessing and feature extraction after adjusting appropriate parameters.
- It includes five steps: (1) Tokenization (2) Noise Removal (3) Stop words Removal (4) Capitalization Conversion (5) Stemming.
- Feature selection methods: word2vec, fastText, and Glove.

Step 2: Training classifiers.

- Run `Classifiers_ML.py` to train machine learning-based classifiers.
- Run `Classifiers_DL_classifiers.py` to train deep learning-based classifiers.
- Machine learning algorithms: Support Vector Machine (SVM), Logistic Regression (LR), Decision Tree (DT), Bernoulli Naive Bayes (NB), and k-Nearest Neighbor (kNN).
- Deep learning algorithm: TextCNN

Step 3: Ensemble classifier.

- Run `Ensemble classifier.py` to conduct voting strategy. -->


## Experiment Environment

requirements:
- `torch==1.11.0`
- `numpy==1.22.3`
- `gensim==4.1.2`
- `fasttext==0.9.2`
- `pandas==1.4.1`
- `torchtext==0.12.0`
- `sklearn==0.0`
- `scikit-learn==1.0.2`
- `w2vembeddings==0.1.2`
- `matplotlib==3.5.1`
- `tqdm==4.62.3`
- `nltk==3.7`

<!-- ## Cite

```
@article{Li2023vsae,
  author = {Li, Ruiyin and Avgeriou, Paris and Liang, Peng},
  title = {Towards Automatic Identification of Violation Symptoms of Architecture Erosion: Experiments and Industrial Validation},
  journal = {IEEE Transactions on Software Engineering},
  year = 2023,
  month = {},
  volume = ,
  number = ,
  issn = {},
  doi = {},
``` -->
