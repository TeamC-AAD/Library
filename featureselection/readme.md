
## Problem Statement

In this following selection , we have applied Genetic algorithm in the field of machine Learning. Given a set of feaures <i>n</i> and given the condition that we must selection <i>m</i> features from these <i>n</i> features such that the linear regression prediction of the test set using these features result in the least possible mean squared errors. The details of the datasets and the Libraries (non-trivial) used are follows:

- <b>Dataset Used</b> : <a href="http://archive.ics.uci.edu/ml/datasets/communities+and+crime"> Communities and Crime Dataset, UCI </a>
- <b>Libraries Used</b> : Pandas, Scikit-Learn

## Approach

We used Genetic Algorithms to solve this problem. The walkthrough of our code has been provided below:

<b>Step 1</b>: Import the libraries

//Embed f2

<b>Step 2</b>: Preprocess the data

//Embed f3,f4

<b>Step 3</b>: Define fitnesss function. The fitness function for a given set of chromosomes is defined as follows:

<img src="http://www.sciweavers.org/upload/Tex2Img_1605900223/render.png">
<img src="http://www.sciweavers.org/upload/Tex2Img_1605900540/render.png">

//Embed f5
 
<b>Step 4</b>: Run using our Library! The results after a few convergences in generations have been shown below:

<div id="#f6"></div>

<img src="https://i.ibb.co/LQJ8Ht6/Screen-Shot-2020-11-20-at-20-12-22.png" alt="Screen-Shot-2020-11-20-at-20-12-22" border="0">
