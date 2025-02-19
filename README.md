# OWMM-VLM

This repo maintains an overview of the OWMM-VLM project, as introduced in paper "Open World Mobile Manipulation with End-to-End Visual Langauge Model".

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Install Habitat environment and datasets](#install-habitat-environment-and-datasets)
  - [Install VLM dependencies](#install-vlm-dependencies)
- [Usage](#usage)
- [Credit](#credit)

## Introduction

**OWMM-VLM** presents the first VLM for the OWMM task, which simultaneously enables 1) scene-level multimodal understanding, 2) ego-centric decision making and 3) generation of executable action sequences with spatial coordinates in single model. With its end-to-end learning fashion and data-driven nature, it allows scaling up and enhances the robustness of the system.

In this repository, we provide the complete pipeline code for data collection and data annotation, as well as the code for step evaluation and simulator evaluation.

## Installation

You should first clone our repo:

```bash
git clone https://github.com/owmm-vlm-project/OWMM-VLM.git
```

### Install Habitat environment and datasets

Please follow the instructions in the [Install Habitat Environment](./install_habitat.md) to install the Habitat environment. Please refer to the Meta official repository [habitat-lab](https://github.com/facebookresearch/habitat-lab) for troubleshooting and more information. 

For extra dependencies in Habitat and original datasets used in OWMM-VLM, please follow the instructions in [Habitat-MAS Package](./dataset_generation/habitat-lab/habitat-mas/README.md).

### Install VLM dependencies

For the dependencies required for model fine-tuning and deployment, please refer to [InternVL2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/). For the dependencies of the baselines, please refer to the dependency downloads of [PIVOT](https://huggingface.co/spaces/pivot-prompt/pivot-prompt-demo/tree/main), [Robopoint](https://github.com/wentaoyuan/RoboPoint), and [GPT](https://platform.openai.com/docs/quickstart).

## Usage

For dataset generation and simulator evaluation, Please follow the instructions in [dataset_generation](./dataset_generation/README.md). After sampling dataset from dataset generation, please refer to the instructions in [dataset_annotation](./dataset_annotation/README.md) to obtain annotated datasets. For step evaluation, please follow the instructions in [step_evaluation](./dataset_evalutation/README.md).

## Credit

This repo is built upon [EMOS](https://github.com/SgtVincent/EMOS), which built based on the [Habitat Project](https://aihabitat.org/) and [Habitat Lab](https://github.com/facebookresearch/habitat-lab) by Meta. 
We would like to thank the authors of EMOS and the original Habitat project for their contributions to the community.
