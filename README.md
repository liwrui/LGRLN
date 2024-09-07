# Language-Guided Graph Representation Learning for Video Summarization

<div align="center">

[**Wenrui Li**](https://liwrui.github.io/),
**Wei Han**,
**Hengyu Man**,
[**Wangmeng Zuo**](https://scholar.google.com/citations?user=rUOpCEYAAAAJ),
[**Xiaopeng Fan**](https://scholar.google.cz/citations?hl=zh-CN&user=4LsZhDgAAAAJ&view_op=list_works&sortby=pubdate)
[**Yonghong Tian**](https://scholar.google.cz/citations?user=fn6hJx0AAAAJ&hl=zh-CN),


</div>


## Introduction
With the rapid growth of video content on social media platforms, video summarization has become a crucial task in multimedia processing. However, existing methods encounter significant challenges in capturing global dependencies in video content and accommodating multimodal user customization. Moreover, temporal proximity between video frames does not always correspond to semantic proximity. To tackle these challenges, we propose a novel Language-guided Graph Representation Learning Network (LGRLN) for video summarization. Specifically, we introduce a video graph generator that converts video frames into a structured graph to preserve temporal order and contextual dependencies. By constructing forward, backward, and undirected graphs, the video graph generator effectively preserves the sequentiality and contextual relationships of video content. We designed an inner graph relationship reasoning module with a dual-threshold graph convolution mechanism. This mechanism distinguishes semantically relevant frames from irrelevant ones by calculating cosine similarity between nodes, thereby intelligently filtering and aggregating information from adjacent frames. Additionally, our proposed language-guided cross-modal embedding module integrates user-provided language instructions into video sequences, generating personalized video summaries that align with specific textual descriptions. To resolve the one-to-many mapping problem in video summarization, we model the output of the summary generation process as a mixture Bernoulli distribution and solve it using the EM algorithm, accommodating the diverse annotation strategies employed by different annotators for the same video. Finally, we introduce a bi-threshold cross-entropy loss function to manage varying annotations from different annotators for the same video. Experimental results show that our method outperforms existing approaches across multiple benchmarks, particularly excelling in handling multimodal tasks. Moreover, we proposed LGRLN is more suitable for real-world applications, as it reduces inference time and model parameters by 87.8\% and 91.7\%, respectively.

## Codes
Coming Soon in later septemper 2024!

## Run

### init project
+ download this project by
  ```git clone https://github.com/liwrui/LGRLN.git```
+ download TVSum and SumMe following https://github.com/IntelLabs/GraVi-T/blob/main/docs/GETTING_STARTED_VS.md
+ download VideoXum from https://huggingface.co/datasets/jylins/videoxum
+ if want to download downloaded parameters, place them to ./results/
+ the final struture should be:
  ```
  |-- LGRLN
    |-- configs
      |-- SumMe
        |-- SPELL_default.yaml
      |-- TVSum
        |-- SPELL_default.yaml
      |-- VideoXum
        |-- SPELL_default.yaml
    |-- data
      |-- annotations
        |-- SumMe
          |-- eccv16_dataset_summe_google_pool5.h5
        |-- TVSum
          |-- eccv16_dataset_tvsum_google_pool5.h5
        |-- videoxum
          |-- blip
          |-- test_videoxum.json
          |-- train_videoxum.json
          |-- val_videoxum.json
          |-- vt_clipscore
      |-- graphs
      |-- generate_temporal_graphs.py
    |-- gravit
    |-- results
      |-- SPELL_VS_SumMe_default
      |-- SPELL_VS_TVSum_default
      |-- SPELL_VS_VideXum_default
    |-- tools
  ```
### pretreat SumMe and TVSum
+ run
    ```python data/generate_temporal_graphs.py --dataset SumMe --features eccv16_dataset_summe_google_pool5 --tauf 10 --skip_factor 0```
    ```python data/generate_temporal_graphs.py --dataset TVSum --features eccv16_dataset_tvsum_google_pool5 --tauf 5 --skip_factor 0```
+ videoxum don't need pretreatment

### training
+ if just want to eval, download parameters first and skip this chaptar
+ train on SumMe and TVSum
    ```python tools/train.py --cfg configs/SumMe/SPELL_default.yaml --split 4```
    ```python tools/train.py --cfg configs/TVSum/SPELL_default.yaml --split 4```
+ train on VideoXum
    ```python tools/train_videoxum.py --cfg configs/VideoXum/SPELL_default.yaml --split 4```

### eval
  | dataset | f1 | tau  |  rho  |
  | --- | ----- | ---- |  -----  |
  |  SumMe | 54.7   | 0.14 | 0.19 |
  | TVSum   | 58.3  | 0.30 | 0.43 |
  | VideoXum | 32.1 | 0.19 | 0.26 |
  ```python tools/eval.py --exp_name SPELL_VS_SumMe_default --eval_type VS_max --split 4```
  ```python tools/eval.py --exp_name SPELL_VS_TVSum_default --eval_type VS_avg --split 4```
  ```python tools/eval_videoxum.py --exp_name SPELL_VS_VideoXum_default --eval_type VS_avg --split 4```


### bugs
+ if encountered an error with ./gravit/, modify that folder from https://github.com/IntelLabs/GraVi-T/tree/main/gravit





