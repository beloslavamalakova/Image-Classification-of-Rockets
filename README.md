# Cluster Vs Grad bombs analysis

### Aim of the project
#### Testing algorithms for classification of a variety of rockets. We are using Logistic regression in order to derive grad rockets form cluster munitions. The theme was inspired by the recent military actions on the territory of the sovereign state of Ukraine, and the usage of illegal weapons and munitions.

### Used data
#### We are using images of Cluser munitions and Grad rockets. The cluster munitions are a form of air-dropped or ground-launched explosive weapon that release or eject smaller submunitions. Commonly, this is a cluster bomb that ejects explosive bomblets that are designed to kill personnel and destroy vehicles. Other cluster munitions are designed to destroy runways or electric power transmission lines, disperse chemical or biological weapons, or to scatter land mines. Some submunition-based weapons can disperse non-munitions, such as leaflets. The BM-21 "Grad" is a Soviet truck-mounted 122 mm multiple rocket launcher. The weapons system and the M-21OF rocket were first developed in the early 1960s, and saw their first combat use in March 1969 during the Sino-Soviet border conflict.BM stands for boyevaya mashina (Russian: боевая машина – combat vehicle), and the nickname grad means "hail". The complete system with the BM-21 launch vehicle and the M-21OF rocket is designated as the M-21 field-rocket system. The complete system is more commonly known as a Grad multiple rocket launcher system.

### Data preparation
#### The data has features. Each “feature” represents a pixel in the image, and each pixel can take on any integer value from 0 to 255. A large value for a pixel means that there is writing in that part of the image.We can see a few examples, by plotting the 784 features as a 28x28 grid. In these images, white pixels indicate high values in the feature matrix.
![Data](https://drive.google.com/file/d/1pZqIqrr6W9GfKk-z-msdDuyGOpCh-hR1/view?usp=sharing)


### Data splitting
#### Data splitting is a process used to separate a given dataset into at least two subsets called 'training' (or 'calibration') and 'test' (or 'prediction'). Our dataset has ../test and ../train. At the ../train the data is labeled with respectfully 'grad', 'cluster' at a specific ../.json file. We are using the train dataset in order to train the algorithms with already labeled and seen data, and the test dataset is an unseen dataset used for the validation purposes of experiments.


### Logistic Regression
![logistic regression results](https://drive.google.com/file/d/1MyuFmQzKOzdpe9G6uUxp43cOpoefC_qV/view?usp=sharing)

