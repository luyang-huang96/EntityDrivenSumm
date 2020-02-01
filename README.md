# An Entity Driven Framework for Abstractive Summarization
Code for EMNLP2019 paper: An Entity-Driven Framework for Abstractive Summarization  

This code is adapted from https://github.com/ChenRocks/fast_abs_rl. 






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
