# Ways of splitting data
data(iris) # using inbuild iris data 

# Splitting using sample function 
set.seed(1)
data(iris)
trainIndex = sample(nrow(iris), nrow(iris)*.8)
trainIris1 = iris[trainIndex,]
testIris1 = iris[-trainIndex,]
table(trainIris1$Species)

# Splitting using sample.split, tries a nearest propotion that evenly splits the data
library(caTools)
set.seed(1)
data(iris)
iris$spl = sample.split(iris,SplitRatio = 0.5) 
trainIris2 = subset(iris, iris$spl == TRUE) 
testIris2 = subset(iris, iris$spl == FALSE)
table(trainIris2$Species)

# Splitting using createDataPartition
library(caret)
set.seed(1)
data(iris)
trainIndex = createDataPartition(iris$Species, p = .7, list = FALSE, times = 1)
trainIris3 = iris[ trainIndex,]
testIris3 = iris[-trainIndex,]
table(trainIris3$Species)

# Splitting using dplyr
library(dplyr)
set.seed(1)
data(iris)
iris$id = 1:nrow(iris)
trainIris4 = sample_frac(iris, .8)
testIris4  = anti_join(iris, trainIris4, by = 'id')
table(trainIris4$Species)


