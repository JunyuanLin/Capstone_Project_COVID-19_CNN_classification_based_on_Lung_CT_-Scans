# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project : COVID-19 classification based on Chest CT Scan

##### by: Lin Junyuan, SG-DSI-14

## Introduction
COVID-2019 caused worldwide disruption upon emergence during the end of 2019. Global efforts at containing the virus resulted in an urgent need to be able to test for the infection. This project aims to develop a convolutional neural network (CNN) based model to assist in addressing that need.

## Problem Statement

To develop a Convolutional Neural Network-based classification model, based on Chest CT Scan data in the axial view, to determine whether COVID infection is present, with a targeted performance of **80%** for both **accuracy** and **sensitivity**.

## Executive Summary

The novel coronavirus 2019 (COVID-2019) was first detected in Wuhan, China during the end of December 2019. Rapidly, it spread throughout the world and was classified as a pandemic within months. It was highly disruptive, driving cities and states into lockdowns in a bid to contain its spread. The global economy, lives and livelihoods of millions were adversely affected.

During the early phases, early detection and containment were proven to be highly effective counter-measures. However, being a novel coronavirus, test kits were not readily available. Detection was key and many countries around the world prioritise testing and were actively building up their capabilities in this respect.

Radiological imaging techniques, such as chest CT scans has been found to contain discernable effects of a COVID-19 infection. In these scans, Ground Glass opacities (GGOs) show up in infected patients' scans and expands as the disease's severity increases. Hence, this project proposes to apply Artificial Intelligence (AI) techniques, namely convolutional neural network (CNN) to detect COVID-19. If successful, this technology can be applied to serve as an auxiliary diagnostic tool, or further expanded to trace and track disease severity.

Based on the amount of data available (May 2020), our model was able to attain an accuracy of 0.83, a sensitivity of 0.95 and AUC ROC score of 0.87.

 ## Conclusions and Recommendations

In conclusion, the selected model has an accuracy of 0.83, sensitivity of 0.95 and AUC ROC score of 0.87. The objective, as stated out in the problem statement, has been met.

During the course of this project, the lack of consistent and quality data was an impediment. However, there are multiple efforts globally to build up databases. As more data is gathered, subsequent refinement can be performed on this model to improve its performance further.

Based on some of the literature, the appearance of GGOs is not unique to COVID-19. With more data, a Multi-Class classification utilising the same CNN approach can be developed to order to differentiate between Covid infection and other types of viral/bacterial infection. There is also the possibility of exploring other methodologies such as Transfer Learning or CNNs utilising 3D CT Scan data (i.e. every slice from a .nii file).

## Setup

###### Required libraries

`numpy`	| `pandas` | `matplotlib` | `Opencv` | `nibabel` | `sklearn` | `tensorflow.keras` (not `keras`)

###### Instructions

1. After downloading, download the source data from the provided link and unpack in their respective folders.

## File Navigation

```
Capstone_Project
|__ assets
|   |__ *.jpg           ## images for notebook
|__ checkpoints
|   |__ *.hdf5			## checkpoints from fitting model(s)
|__ code
|	|__ preprocessing.py
|__ data
|	|__ CT_Covid
|		|__ *.jpg		## Covid positive images from source
|	|__ CT_nonCovid
|		|__ aug_test
|			|__ *.jpeg	## Covid negative images from data augmentation
|		|__ aug_train	
|			|__ *.jpeg	## Covid negative images from data augmentation  
|		|__ *.jpg		## Covid negative images from source
|__ logs
|	|__ fit
|		|__ *			## logs folders for tensorboard
|__ models
|	|__ *.h5			## saved models
|__ source
|   |__ S1_covid19-ct-scans					*** unpack source 1 here
|	|__ S2_COVID-CT-master					*** unpack source 2 here
|	|__ S3_covid-chestxray-dataset-master	*** unpack source 3 here
|	|__ S4_covid_19_public_data				*** unpack source 4 here
|	|__ S5_radiopedia						
|	|__ S6_mosmed							*** unpack source 6 here
|__ Capstone Project - COVID-19 classification based on Lung CT Scans.pdf
|__ Main_notebook.ipynb
|__ metadata.csv
|__ results.csv
|__ README.md

```

## Data sources

Source 1: `Kaggle\COVID-19 CT scans` [Link](https://www.kaggle.com/andrewmvd/covid19-ct-scans)

Source 2: `GitHub\UCSD-AI4H\COVID-CT` [Link](https://github.com/UCSD-AI4H/COVID-CT)

Source 3: `GitHub\ieee8023\covid-chestxray-dataset` [Link](https://github.com/ieee8023/covid-chestxray-dataset)

Source 4: `Toy Dataset synthesized from actual patient scans from Covid19Challenge.eu` [Link](https://www.covid19challenge.eu/covid-19-toy-datasets-released/)

Source 5: `Scraped CT images from Radiopaedia.org` [Link](https://radiopaedia.org/)

Source 6: `MosMedData: COVID19_1000 Dataset` [Link](https://mosmed.ai/en/)

## References

- Lung Cancer Detection and Classification based on Image Processing and Statistical Learning [Link](https://arxiv.org/pdf/1911.10654.pdf)
- GitHub\COVID-Net Open Source Initiative [Link](https://github.com/lindawangg/COVID-Net)
- Automated detection of COVID-19 cases using deep neural networks with X-ray images [Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7187882/)
- Classification of COVID-19 patients from chest CT images using multi-objective differential evolution–based convolutional neural networks [Link](https://link.springer.com/article/10.1007/s10096-020-03901-z)
- Diagnosing COVID-19 Pneumonia from X-Ray and CT Images using Deep Learning and Transfer Learning Algorithms [Link](https://www.researchgate.net/publication/340374481_Diagnosing_COVID-19_Pneumonia_from_X-Ray_and_CT_Images_using_Deep_Learning_and_Transfer_Learning_Algorithms)