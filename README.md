# TAG
Implementation of [CoVeR](https://arxiv.org/abs/1802.07839) (Covariate-Specific Vector Representations with Tensor Decomposition), a model that learn word embeddings and a series of covariances to transform normal embeddings to covariate-specific. This is a re-implementation of [GloVe](http://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation) proposed by Pennington J., Socher R. and Manning C.D. (2014)

I used this model to analyse the difference on word usage between male and female speech in Old Bailey Voice Corpus from 1780 to 1880, which thus suggesting different meanings. You will need `tf_glove.py` from Simon G. (2017) before running `CoVerModel.py`, you can get it from:

https://github.com/GradySimon/tensorflow-glove

`CoVerModel.py` -- actual code implementing CoVeR model (Tian et al., 2018)

`main.py` -- example usage of `CoVerModel.py` on Old Bailey Voice Corpus separated in female and male corpora

## Prerequisite
- v1.0.0 tensorflow
- v0.20.1 pandas
- v2.0.11 spacy with `eng_core_web_md` installed in v2.0.0
- v1.15.0 numpy
- v0.19.2 scikit-learn

## Data set
You can find the Old Bailey Voice Corpus here:

https://github.com/sharonhoward/voa

The file I used in `main.py` is the unzipped version of `OBV2/obv_words_v2_28-01-2017.tsv.zip`. It is the updted verion on May 2017.

## How to use
```python
>>> from CoVerModel import CoVerModel
>>> cover = CoVeRModel(embedding_size=300, context_size=10)
>>> cover.fit_corpora(corpora)
>>> cover.train()
>>> cover.embeddings
array([[-0.97056437, -0.4124898 ,  0.74619746, ...,  2.2409537 ,
        -2.3252304 ,  0.0969246 ],
       [-0.6628446 , -0.92629707,  2.0215068 , ...,  1.9445996 ,
        -1.5850129 , -0.1194554 ],
       ...,
       [-0.72753906,  0.49558735,  1.2009518 , ..., -0.04062271,
        -0.49638033,  0.16031623]], dtype=float32)
>>> cover.covariates
array([[ 5.10783792e-01,  1.36624873e-01,  2.43794426e-01,
        ...
        -3.18439603e-01, -3.65320116e-01, -5.38883567e-01],
       [ 1.18549293e-09, -2.79872656e-01, -2.56543934e-01,
        ...,
         2.83558279e-01, -3.57488275e-01, -6.27053738e-01]], dtype=float32)
>>> cover.generate_tsne()
```
![female_full](https://user-images.githubusercontent.com/28641434/44266037-ee920400-a220-11e8-8931-58706158bff4.png)

## Parameters

Parameters in creating a CoVeRModel is more or less the same as `tf_glove.py`, with `embedding_size` and `context_size` as complusory arguments

The format of the corpora required for `fit_corpora` are multidimensional iterables of strings, where strings are tokens as below:

```
[
  [["This","is","a","sentence","in","1st","corpus","."],
    ["Second","sentence","in","1st","corpus","."]],

  [["Sentence","1","in","2nd","corpus:","."],
   ["Sentence","2","in","2nd","corpus","."]]
]
 ```
For full implementation with Old Bailey Voice Corpus, see `main.py`.

## References

- Pennington J., Socher R. and Manning C.D., ACL (2014). [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162).
- Tian K., Zhang T. and Zou J., ArXiV (2018). [CoVeR: Learing Covariate-Specific Vector Representations with Tensor Decompositions](https://arxiv.org/pdf/1802.07839.pdf).
- Simon, G. (2017). tensorflow-glove, GitHub Repository [online] Available at: https://github.com/GradySimon/tensorflow-glove