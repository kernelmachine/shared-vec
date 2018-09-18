# shared-vec
Project word vectors into shared space with linear CCA

This project follows Faruqui and Dyer 2010 (http://www.aclweb.org/anthology/E14-1049) for creating shared representations of word vectors.

This code can be used, for example, to create multilingual word vectors, where word vectors in target languages (e.g. French, Tagalog, Hindi) can be projected into a source language space, like English. This would allow one to share model parameters or corpora across languages.

To project a set of word vectors into another word vector space, you need:

1) `dict.txt` - a dictionary aligning source and target words (does not need to be exhaustive, but the more alignment examples the better). Each line of this file should be of format `SRC_WORD ||| TGT_wORD\n`.

2) `src.vecs` - word vectors in the source domain. Each line of this file should be of format `WORD\tVECTOR\n` where `WORD` is a string of characters, and `VECTOR` is a space-separated set of floats. This is the canonical format of word2vec files, fastText files, etc.

3) `tgt.vecs` - word vectors in the target domain. Each line of this file should be of format `WORD\tVECTOR\n` where `WORD` is a string of characters, and `VECTOR` is a space-separated set of floats. This is the canonical format of word2vec files, fastText files, etc.


To get example dictionaries and word vectors, head over to https://github.com/facebookresearch/MUSE


Then, clone this repo and run


```
$ python project-vectors.py --alignment dict.txt --src src.vecs --tgt tgt.vecs --out tgt.src.vecs
```

This will create a file of word vectors from the target domain projected into the source domain's space.
