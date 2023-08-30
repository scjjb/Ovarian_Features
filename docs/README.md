## Predicting Ovarian Cancer Treatment Response in Histopathology using Hierarchical Vision Transformers and Multiple Instance Learning 
<img src="CISTIB logo.png" align="right" width="240"/>

*HIPT-ABMIL is a transformer-based approach to classifying histopathology slides which leverages spatial information for better prognostication.* 

<img src="HIPT-AMBIL-ModelDiagram-Background-min.png" align="centre" width="900"/>

This repo was made as part of an entry to the MICCAI 2023 challenge [*automated prediction of treatment effectiveness in ovarian cancer using histopathological images*](https://github.com/cwwang1979/MICCAI_ATEC23challenge).

HIPT-CLAM is a multiple instance learning (MIL) approach in which features are extracted from 4096x4096 pixel regions using the pretrained hierarchical transformer model [HIPT_4K](https://github.com/mahmoodlab/HIPT) and these features are aggregated to generate a slide-level representation using the attention-based multiple instance learning (ABMIL) approach [CLAM](https://github.com/mahmoodlab/CLAM). 




## Reference
This code is an extension of our [previous repository](), which itself was forked from the [CLAM repository](https://github.com/mahmoodlab/CLAM) with corresponding [paper](https://www.nature.com/articles/s41551-020-00682-w). Code is also used from the [HIPT repository](https://github.com/mahmoodlab/HIPT), including pretrained model weights. This repository and the original CLAM repository are both available for non-commercial academic purposes under the GPLv3 License.
