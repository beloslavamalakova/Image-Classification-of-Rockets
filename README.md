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
#### We will use sklearn's LogisticRegression. Unlike the linear regression, there is no closed form solution to the least squares parameter estimate in logistic regression. Therefore, we need to use a “solver” which finds a numerical solution. Several solvers are available for use with sklearn's LogisticRegression, but they don't all support all varieties of logistic regression. We will use the saga solver, which works well when there is a large number of samples, supports logistic regression with no regularization penalty, L1 penalty, L2 penalty, or ElasticNet (which uses both penalties), and also supports multinomial regression with multiple classes, using the softmax function. 
#### One benefit of logistic regression is its interpretability — we can use the coefficient values to understand what features (i.e. which pixels) are important in determining what class a sample belongs to.The following plot shows the coefficient vector for each class, with positive coefficients in blue and negative coefficients in red.
#### We can see which pixels are positively associated with belonging to the class, and which pixels are negatively associated with belonging to the class.For example, consider Class 0. If a sample has large values in the pixels shown in blue (the 0 shape around the center of the image), the probability of that sample being a 0 digit increases. If the sample has large values in the pixels in the center of the image, the probability of the sample being a 0 digit decreases.Many pixels have coefficients whose magnitude are very small. These are shown in white, and they are not very important for this classification task.
#### In general, to get the predicted label, we can find the class with the highest probability. If this matches the actual label for the first test sample, then our prediction is correct.
#### Let’s look at our example test point, and compare to our own computations.We use the predict function to predict a label for each sample in the test set. 
#### Now we have to invert the image to match the training data and then we adjust the contrast and scale of the image. Finally, reshape to (1, 784) — 1 sample, 784 features. Now we use the logistic regression model to predict our input image and plot the conditional probability. Here are the results of the classification. The validation accuracy is 0.868421052631579.
![logistic regression results](https://drive.google.com/file/d/1MyuFmQzKOzdpe9G6uUxp43cOpoefC_qV/view?usp=sharing)

