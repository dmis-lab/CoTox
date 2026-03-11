# CoTox: Chain-of-Thought-Based Molecular Toxicity Reasoning and Prediction

[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](https://arxiv.org/abs/2508.03159)
[![IEEE](https://img.shields.io/badge/IEEE-paper-blue)]([IEEE_URL](https://ieeexplore.ieee.org/document/11356816))

![img](./_figure/cotox_figure.jpg)

## Introduction
Can LLM assess molecular toxicity? 💊💀
During drug development, it is crucial to identify whether the chemical compound is toxic or not.
We introduce CoTox, a novel framework that utilizes LLMs for Molecular Toxicity Prediction.
Unlike traditional models that rely solely on molecular structure, CoTox integrates chemical structures, biological pathways, and GO terms to predict six types of organ-specific toxicities, including cardiotoxicity, hepatotoxicity, and nephrotoxicity.
By using Chain-of-Thought prompting, CoTox generates step-by-step reasoning for each prediction, offering transparent and interpretable explanations for why a compound might be toxic.
Interestingly, we also found that IUPAC names work better than SMILES when interfacing with LLMs, thanks to their human-readable format.
Our findings position CoTox as an interpretable and practical tool for early-stage drug development.
