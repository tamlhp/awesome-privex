# Awesome Privacy-Preserving Explainable AI (PrivEx, PPXAI)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Website](https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Official%20Website&up_color=green&up_message=online&url=https%3A%2F%2Fawesome-privex.github.io%2F)](https://awesome-privex.github.io/)
![GitHub stars](https://img.shields.io/github/stars/tamlhp/awesome-privex?color=yellow&label=Stars)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Ftamlhp%2Fawesome-privex%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/>



A collection of academic articles, published methodology, and datasets on the subject of **Privacy-Preserving Explainable AI**.

- [A Survey of Privacy-Preserving Model Explanations: Privacy Leaks, Attacks, and Countermeasures](#awesome-privex)
  - [Surveys](#existing-surveys)
  - [Approaches](#approaches)
  - [Datasets](#datasets)
    - [Type: Image](#type-image)
    - [Type: Tabular](#type-tabular)
    - [Type: Graph](#type-graph)
    - [Type: Text](#type-text)
  - [Evaluation Metrics](#evaluation-metrics)

A sortable version is available here: https://awesome-privex.github.io/

----------

## Existing Surveys
| **Paper Title** | **Venue** | **Year** | 
| --------------- | ---- | ---- | 
| [A Survey of Privacy Attacks in Machine Learning](https://dl.acm.org/doi/full/10.1145/3624010) | _CSUR_ | 2023 |
| [SoK: Taming the Triangle -- On the Interplays between Fairness, Interpretability and Privacy in Machine Learning](https://arxiv.org/abs/2312.16191) | _arXiv_ | 2023 |

----------

## Approaches

| **Title** | **Year** | **Venue** | **Target Explanations** | **Attacks** | **Defenses** | **Code** |
| --------------- | :----: | ---- | :----: | :----: | :----: | :----: |
| [Please Tell Me More: Privacy Impact of Explainability through the Lens of Membership Inference Attack](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a120/1Ub23teQ7PG) | 2024 | _SP_ | Feature-based | Membership Inference | Differential Privacy, Privacy-Preserving Models, DP-SGD | - |
| [On the Privacy Risks of Algorithmic Recourse](https://proceedings.mlr.press/v206/pawelczyk23a.html) | 2023 | _AISTATS_ | Counterfactual | Membership Inference | Differential Privacy | - |
| [The Privacy Issue of Counterfactual Explanations: Explanation Linkage Attacks](https://dl.acm.org/doi/full/10.1145/3608482) | 2023 | _TIST_ | Counterfactual | Linkage | Anonymisaion | - |
| [Feature-based Learning for Diverse and Privacy-Preserving Counterfactual Explanations](https://dl.acm.org/doi/abs/10.1145/3580305.3599343) | 2023 | _KDD_ | Counterfactual | - | Perturbation | [[Code]](https://github.com/isVy08/L2C/) |
| DP-XAI | 2023 | _Github_ | ALE plot | - | Differential Privacy | [[Code]](https://github.com/lange-martin/dp-global-xai) |
| [Membership Inference Attack on Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9750191) | 2023 | _TPS-ISA_ | Feature-based | Graph Extraction | Perturbation | [[Code]](https://github.com/iyempissy/graph-stealing-attacks-with-explanation) |
| [Probabilistic Dataset Reconstruction from Interpretable Models](https://arxiv.org/abs/2308.15099) | 2023 | _arXiv_ | Interpretable Surrogates | Data Reconstruction | - | [[Code]](https://github.com/ferryjul/ProbabilisticDatasetsReconstruction) |
| [DeepFixCX: Explainable privacy-preserving image compression for medical image analysis](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1495) | 2023 | _WIREs-DMKD_ | Case-based | Identity recognition | Anonymisation | [[Code]](https://github.com/adgaudio/DeepFixCX) |
| [XorSHAP: Privacy-Preserving Explainable AI for Decision Tree Models](https://eprint.iacr.org/2023/1859) | 2023 | _Preprint_ | Shapley | - | Multi-party Computation | - |
| [Inferring Sensitive Attributes from Model Explanations](https://dl.acm.org/doi/abs/10.1145/3511808.3557362) | 2022 | _CIKM_ | Gradient-based, Perturbation-based | Attribute Inference | - | [[Code]](https://github.com/vasishtduddu/AttInfExplanations) |
| [Model explanations with differential privacy](https://dl.acm.org/doi/abs/10.1145/3531146.3533235) | 2022 | _FAccT_ | Feature-based | - | Differential Privacy | - |
| [Evaluating the privacy exposure of interpretable global explainers](https://ieeexplore.ieee.org/abstract/document/10063510/), [Privacy Risk of Global Explainers](https://ebooks.iospress.nl/doi/10.3233/FAIA220206) | 2022 | _CogMI_ | Interpretable Surrogates | Membership Inference | - | - |
| [Privacy-Preserving Case-Based Explanations: Enabling Visual Interpretability by Protecting Privacy](https://ieeexplore.ieee.org/document/9729808/) | 2022 | _IEEE Access_ | Example-based | - | Anonymisation | - |
| [On the amplification of security and privacy risks by post-hoc explanations in machine learning models](https://arxiv.org/abs/2206.14004) | 2022 | _arXiv_ | Feature-based | Membership Inference | - | - |
| [Differentially Private Shapley Values for Data Evaluation](https://arxiv.org/abs/2206.00511) | 2022 | _arXiv_ | Shapley | - | Differential Privacy | [[Code]](https://github.com/amiratag/DataShapley) |
| [On the Privacy Risks of Model Explanations](https://dl.acm.org/doi/abs/10.1145/3461702.3462533) | 2021 | AIES | Feature-based, Shapley, Counterfactual | Membership Inference | - | - |
| [MEGEX: Data-Free Model Extraction Attack against Gradient-Based Explainable AI](https://arxiv.org/abs/2107.08909) | 2021 | _arXiv_ | Gradient-based | Model Extraction | - | [[Code]](https://github.com/cake-lab/datafree-model-extraction) |
| [Robust Counterfactual Explanations for Privacy-Preserving SVM](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1581005&dswid=5229), [Robust Explanations for Private Support Vector Machines](https://arxiv.org/abs/2102.03785) | 2021 | _ICML-Workshop_ | Counterfactual | - |  Private SVM | [[Code]](https://github.com/rami-mochaourab/robust-explanation-SVM) |
| [When Differential Privacy Meets Interpretability: A Case Study](https://arxiv.org/abs/2106.13203) | 2021 | _RCV-CVPR_ | Interpretable Models | - | Differential Privacy | - |
| [Differentially Private Quantiles](https://proceedings.mlr.press/v139/gillenwater21a.html) | 2021 | _ICML_ | Quantiles | - | Differential Privacy | [[Code]](https://github.com/google-research/google-research/tree/master/dp_multiq) |
| [Privacy-preserving generative adversarial network for case-based explainability in medical image analysis](https://ieeexplore.ieee.org/abstract/document/9598877/) | 2021 | _IEEE Access_ | Example-based | - | Generative Anonymisation | - |
| [Interpretable and Differentially Private Predictions](https://ojs.aaai.org/index.php/AAAI/article/view/5827) | 2020 | _AAAI_ | Locally linear maps | - | Differential Privacy | [[Code]](https://github.com/frhrdr/dp-llm) |
| [Model extraction from counterfactual explanations](https://arxiv.org/abs/2009.01884) | 2020 | _arXiv_ | Counterfactual | Model Extraction | - | [[Code]](https://github.com/aivodji/mrce) |
| [Model Reconstruction from Model Explanations](https://dl.acm.org/doi/10.1145/3287560.3287562) | 2019 | _FAT*_ | Gradient-based | Model Reconstruction, Model Extraction | - | - |
| [Interpret Federated Learning with Shapley Values](https://arxiv.org/abs/1905.04519) | 2019 | __ |  Shapley | - | Federated | [[Code]](https://github.com/crownpku/federated_shap) |
| [Model inversion attacks that exploit confidence information and basic countermeasures](https://dl.acm.org/doi/abs/10.1145/2810103.2813677) | 2015 | _CCS_ | Confidence scores | Reconstruction, Model Inversion | - | - |


----------
**Disclaimer**

Feel free to contact us if you have any queries or exciting news on machine unlearning. In addition, we welcome all researchers to contribute to this repository and further contribute to the knowledge of machine unlearning fields.

If you have some other related references, please feel free to create a Github issue with the paper information. We will glady update the repos according to your suggestions. (You can also create pull requests, but it might take some time for us to do the merge)


[![HitCount](https://hits.dwyl.com/tamlhp/awesome-privex.svg?style=flat-square)](http://hits.dwyl.com/tamlhp/awesome-privex)
 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=tamlhp.awesome-privex)
