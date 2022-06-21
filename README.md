# UnsupervisedBotClassification

## README
This repo houses the model and run files for the empirical example in Roman, Brandt, and Miller (2022). "latent_class_bots.R" is the run file, start there. Below is the abstract which provides the motivation for the model.

## Abstract
Behavioral scientists have become increasingly reliant on online survey platforms such as Amazon's Mechanical Turk (Mturk). These platforms have many advantages, for example it provides ease of access to difficult to sample populations, a large pool of participants, and an easy to use implementation. A major drawback is the existence of bots that are used to complete online surveys for financial gain. These bots contaminate data and need to be identified in order to draw valid conclusions from data obtained with these platforms. In this article, we will provide a Bayesian latent class joint modeling approach that can be routinely applied to identify bots and simultaneously estimate a model of interest. This method can be used to separate the bots' response patterns from real human responses that were provided in line with the item content. The model has the advantage that it is very flexible and is based on plausible assumptions that are met in most empirical settings. We will provide a simulation study that investigates the performance of the model under several relevant scenarios including sample size, proportion of bots, and model complexity. We will show that ignoring bots will lead to severe parameter bias whereas the Bayesian latent class model results in unbiased estimates and thus controls this source of bias. We will illustrate the model and its capabilities with data from an empirical political ideation survey with known bots. We will discuss the implications of the findings with regard to future data collection via online platforms.
