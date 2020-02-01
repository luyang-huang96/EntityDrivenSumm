# An Entity Driven Framework for Abstractive Summarization
Code for EMNLP2019 paper: An Entity-Driven Framework for Abstractive Summarization  

This code is adapted from https://github.com/ChenRocks/fast_abs_rl. 

We are not allowed to share data/outputs on New York Times Dataset. If you need data/outputs on New York Times Dataset, please email me with your license and we're glad to share our processed data/outputs on NYT dataset for research purpose.  

My permenant email address: luyang.huang96@gmail.com

## How to train our model  

1. our processed data with gold extract labels and entities can be found here:  

https://drive.google.com/file/d/1PiUqGxIZ2veBzGo1MzisFWaR0mRsBdL3/view?usp=sharing  

2. To train our best model:  

0) specify data path  
`export DATA=[path/to/decompressed/data]`

1). pretrained a *word2vec* word embedding
```
python train_word2vec.py --path=[path/to/word2vec]
```

2) train entity-aware content selection component  
```
python train_extractor_ml.py --path=[path/to/extractor/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin] 
```

3) train our abstract generation component  
```
python train_abstractor.py --path=[path/to/abs_ml/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin] 
```
```
python train_abstractor_rl.py --path=[path/to/abs_rl/model] --abs_dir=[path/to/abs_ml/model] --w2v=[path/to/word2vec/word2vec.128d.226k.bin] [--apposition] (our apposition reward) [--anaphora] (our Pronominal Referential Clarity reward) [--coherence] (our coherence reward) [--all_local] (combine all rewards)
```

4) connect two components  
```
python train_full_rl.py --path=[path/to/joint/model] --abs_dir=[path/to/abs_rl/model] --ext_dir=[path/to/extractor/model]
```

5) decode 
```
python decode_full_model.py --path=[path/to/save/decoded/files] --model_dir=[path/to/joint/model] --beam=[beam_size] [--test/--val]
```

6) evaluate  
```
python eval_full_model.py --[rouge/meteor] --decode_dir=[path/to/save/decoded/files]
```





## Dependencies  
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) >=0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)







If you find our code and paper useful to your research, please consider cite our paper:  
```
@inproceedings{sharma-etal-2019-entity,
    title = "An Entity-Driven Framework for Abstractive Summarization",
    author = "Sharma, Eva  and
      Huang, Luyang  and
      Hu, Zhe  and
      Wang, Lu",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1323",
    doi = "10.18653/v1/D19-1323",
    pages = "3280--3291",
    abstract = "Abstractive summarization systems aim to produce more coherent and concise summaries than their extractive counterparts. Popular neural models have achieved impressive results for single-document summarization, yet their outputs are often incoherent and unfaithful to the input. In this paper, we introduce SENECA, a novel System for ENtity-drivEn Coherent Abstractive summarization framework that leverages entity information to generate informative and coherent abstracts. Our framework takes a two-step approach: (1) an entity-aware content selection module first identifies salient sentences from the input, then (2) an abstract generation module conducts cross-sentence information compression and abstraction to generate the final summary, which is trained with rewards to promote coherence, conciseness, and clarity. The two components are further connected using reinforcement learning. Automatic evaluation shows that our model significantly outperforms previous state-of-the-art based on ROUGE and our proposed coherence measures on New York Times and CNN/Daily Mail datasets. Human judges further rate our system summaries as more informative and coherent than those by popular summarization models.",
}
```
