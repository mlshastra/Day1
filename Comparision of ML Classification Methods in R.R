#performance measure of multile classification methods on the same cancer diagnosis data
library(gmodels) # 

set.seed(1)
data = read.csv('wisc_bc_data.csv') # Read csv file
data2 = data[-1] # take out first id column 

str(data2)
data2$diagnosis = factor( data2$diagnosis) # factorize diagnosis column

data2[2:31] = scale(data2[2:31]) # normalize/scale all columns except diagnosis column

library(caret)
trainindex = createDataPartition(data2$diagnosis, p = .7, list = FALSE, times = 1)

# 1 prepare training sets and apply Naviebayes

data_train_nb = data2[trainindex,]
data_test_nb = data2[-trainindex,]

library(e1071)
library(caTools)

model_nbayes_train = naiveBayes(data_train_nb[,2:31], data_train_nb[,1])

#model_nbayes_train

model_nbayes_test = predict(model_nbayes_train, data_test_nb)

#NavieBayes Method
CrossTable(data_test_nb$diagnosis, model_nbayes_test, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# 2 partioning data for kmeans sets and applying kmeans for 2 clusters, as we know we need only 2 clusters

data_train_km = data2[trainindex,2:31] # Taking data only from variable columns
data_test_km = data2[-trainindex,2:31]

model_kmeans_train = kmeans(data_train_km,2,iter.max = 10)

model_kmeans_test = kmeans(data_test_km,2, iter.max = 10)

data_train_km$clusterid = model_kmeans_train$cluster # adding cluster id to the data sets
data_test_km$clusterid = model_kmeans_test$cluster

data_train_km$diagnosis = data2[trainindex, 1] # adding back class column to the sets
data_test_km$diagnosis = data2[-trainindex, 1]
 
data_test_km$clusterid = ifelse(data_test_km$clusterid == 1,"B","M") # renaming clusters id to class 

#KMeans Method
CrossTable(data_test_km$diagnosis, data_test_km$clusterid, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))


# 3 partioning data for knn sets 

data_train_knn = data2[trainindex,2:31] # taking only 2 to 31 columns 
data_test_knn = data2[-trainindex,2:31]

data_train_knn_cl = data2[trainindex,1] # taking class as spearate data 
data_test_knn_cl = data2[-trainindex,1]

library(class)

data_knn_test_pred = knn(train = data_train_knn, test = data_test_knn, cl = data_train_knn_cl, k = 10)


table(data_knn_test_pred, data_test_knn_cl)

spineplot(factor(data_knn_test_pred)~factor(data_test_knn_cl), col = c(5,6))

#KNN Method
CrossTable(data_knn_test_pred, data_test_knn_cl, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# 4 partioning data and applying ctree decision tree

library('party')

data_train_ctree = data2[trainindex,]
data_test_ctree = data2[-trainindex,]

feature = paste("feature",2:31,sep = "")

feature

colnames(data_train_ctree) = c("diagnosis", feature)
colnames(data_test_ctree) = c("diagnosis", feature)

  
fmla = as.formula(paste("diagnosis ~ ", paste(feature, collapse = "+")))

fmla

model_data_train_ctree = ctree(fmla , data_train_ctree)

plot(model_data_train_ctree)


#model_data_test_ctree = ctree(fmla , data_test_ctree)
#model_data_test_ctree
#plot(model_data_test_ctree)

model_data_test_ctree = predict(model_data_train_ctree, data_test_ctree)

model_data_test_ctree


# 5 CTree DT Method
CrossTable(data_test_ctree$diagnosis,model_data_test_ctree, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))


# partioning data and applying tree decision tree

library('tree')

data_train_tree = data2[trainindex,]
data_test_tree = data2[-trainindex,]

feature_c = paste("feature",2:31,sep = "")

colnames(data_train_tree) = c("diagnosis", feature_c)
colnames(data_test_tree) = c("diagnosis", feature_c)


fmla <- as.formula(paste("diagnosis ~ ", paste(feature_c, collapse = "+")))

model_data_train_tree = tree(fmla , data = data_train_tree, method = 'class')


model_data_test_tree = predict(model_data_train_tree,data_test_tree, type = 'class')


#Tree DT Method
CrossTable(data_test_tree$diagnosis,model_data_test_tree, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# 6 partioning data and applying rpart decision tree

library('rpart')

data_train_rpart = data2[trainindex,]
data_test_rpart = data2[-trainindex,]

feature_b = paste("feature",2:31,sep = "")

colnames(data_train_rpart) = c("diagnosis", feature_b)
colnames(data_test_rpart) = c("diagnosis", feature_b)


fmla <- as.formula(paste("diagnosis ~ ", paste(feature_b, collapse = "+")))

model_data_train_rpart = rpart(fmla , data = data_train_rpart, method = "class")

model_data_test_rpart = predict(model_data_train_rpart,data_test_rpart, type = 'class')

library('rpart.plot')
rpart.plot(model_data_train_rpart, digits = 3, fallen.leaves = TRUE, type = 3, extra = 101)


#RPart DT Method
CrossTable(data_test_rpart$diagnosis,model_data_test_rpart, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# 7 partioning data and applying c5.0 decision tree

library('C50')

data_train_c5 = data2[trainindex,]
data_test_c5 = data2[-trainindex,]

feature_d = paste("feature",2:31,sep = "")

colnames(data_train_c5) = c("diagnosis", feature_d)
colnames(data_test_c5) = c("diagnosis", feature_d)

matrix_dimension = list(c("B","M"), c("B","M"))
names(matrix_dimension) = c("actual", "predicted")

matrix_dimension

error_cost = matrix(c(0,30,1,0), nrow = 2, dimnames = matrix_dimension  )

error_cost

fmla <- as.formula(paste("diagnosis ~ ", paste(feature_d, collapse = "+")))

model_data_train_c5 = C5.0(fmla , data = data_train_c5, method = 'class', trials = 10, costs = error_cost) # boost performance by increase number of trials to 10


model_data_test_c5 = predict(model_data_train_c5,data_test_c5, type = 'class')

#C5.0 DT Method
CrossTable(data_test_c5$diagnosis,model_data_test_c5, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# 8 partioning data and applying NN classification

library('neuralnet')


data_train_nn = data2[trainindex,]
data_test_nn = data2[-trainindex,]

feature_d = paste("feature",2:31,sep = "")

colnames(data_train_nn) = c("diagnosis", feature_d)
colnames(data_test_nn) = c("diagnosis", feature_d)

data_train_nn$diagnosis = ifelse(data_train_nn$diagnosis == "B", 1, 0)
data_test_nn$diagnosis = ifelse(data_test_nn$diagnosis == "B", 1, 0)

fmla <- as.formula(paste("diagnosis ~ ", paste(feature_d, collapse = "+")))

model_data_train_nn = neuralnet(formula = fmla , data = data_train_nn, hidden = 5) 

plot(model_data_train_nn)

model_data_test_nn = compute(model_data_train_nn, data_test_nn[2:31])

#cor(model_data_test_nn$net.result,model_data_test_nn$diagnosis)

list = model_data_test_nn$net.result

list


predicted_diagnosis = ifelse(model_data_test_nn$net.result < 1.2 & model_data_test_nn$net.result > 0.5, "B","M")

data_test_nn$diagnosis = ifelse(data_test_nn$diagnosis == 1, "B", "M")

#9 Neuralnet Method
CrossTable(data_test_nn$diagnosis,predicted_diagnosis, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# partioning data and applying KSVM vanilladot

library('kernlab')

data_train_ksvm = data2[trainindex,]
data_test_ksvm = data2[-trainindex,]

feature_d = paste("feature",2:31,sep = "")

colnames(data_train_ksvm) = c("diagnosis", feature_d)
colnames(data_test_ksvm) = c("diagnosis", feature_d)

fmla <- as.formula(paste("diagnosis ~ ", paste(feature_d, collapse = "+")))

model_data_train_ksvm = ksvm(fmla , data = data_train_ksvm,  kernel = "vanilladot") 


model_data_test_ksvm = predict(model_data_train_ksvm,data_test_ksvm)



#10 KSVM vanilladot Method 
CrossTable(data_test_ksvm$diagnosis,model_data_test_ksvm, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# partioning data and applying KSVM rbfdot

library('kernlab')

data_train_ksvm_rbfdot = data2[trainindex,]
data_test_ksvm_rbfdot = data2[-trainindex,]

feature_d = paste("feature",2:31,sep = "")

colnames(data_train_ksvm_rbfdot) = c("diagnosis", feature_d)
colnames(data_test_ksvm_rbfdot) = c("diagnosis", feature_d)

fmla <- as.formula(paste("diagnosis ~ ", paste(feature_d, collapse = "+")))

model_data_train_ksvm_rbfdot = ksvm(fmla , data = data_train_ksvm_rbfdot,  kernel = "rbfdot") 


model_data_test_ksvm_rbfdot = predict(model_data_train_ksvm_rbfdot,data_test_ksvm_rbfdot)

#KSVM rbfdot Method
CrossTable(data_test_ksvm_rbfdot$diagnosis,model_data_test_ksvm_rbfdot, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))



# 11 partioning data and applying Logistical Regression

data_train_logr = data2[trainindex,]
data_test_logr = data2[-trainindex,]

feature_d = paste("feature",2:31,sep = "")

colnames(data_train_logr) = c("diagnosis", feature_d)
colnames(data_test_logr) = c("diagnosis", feature_d)

data_train_logr$diagnosis = ifelse(data_train_logr$diagnosis == "B", 1, 0)
data_test_logr$diagnosis = ifelse(data_test_logr$diagnosis == "B", 1, 0)

fmla <- as.formula(paste("diagnosis ~ ", paste(feature_d, collapse = "+")))

model_data_train_logr = glm(fmla , data = data_train_logr) 

model_data_test_logr = predict(model_data_train_logr,data_test_logr)

library('ROCR') # ROCR takes only continuous data, hence trying this before we change the data

pred = prediction(data_test_logr$diagnosis,data_test_logr$diagnosis)
logr_performance = performance(pred, measure = "tpr", x.measure = "fpr")
plot(logr_performance, colorize = TRUE, text.adj = c(-0.2,1.7))

predicted_diagnosis = ifelse(model_data_test_logr < 1.5 & model_data_test_logr > 0.6, "B","M" ) # Notice that Y is greater than 0.65 instead of .5 

data_test_logr$diagnosis = ifelse(data_test_logr$diagnosis == 1, "B", "M" )

#Log Reg Method
CrossTable(data_test_logr$diagnosis,predicted_diagnosis, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))



#Comparision of knn, kmeans, naivebayes, ctree, rpart, tree, c5


library(gmodels)
#KNN Method
CrossTable(data_knn_test_pred, data_test_knn_cl, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#KMeans Method
CrossTable(data_test_km$diagnosis, data_test_km$clusterid, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#NavieBayes Method
CrossTable(data_test_nb$diagnosis, model_nbayes_test, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#RPart DT Method
CrossTable(data_test_rpart$diagnosis,model_data_test_rpart, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#CTree DT Method
CrossTable(data_test_ctree$diagnosis,model_data_test_ctree, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#Tree DT Method
CrossTable(data_test_tree$diagnosis,model_data_test_tree, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#C5.0 DT Method
CrossTable(data_test_c5$diagnosis,model_data_test_c5, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#Neuralnet Method
CrossTable(data_test_nn$diagnosis,predicted_diagnosis, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#KSVM vanilladot Method 
CrossTable(data_test_ksvm$diagnosis,model_data_test_ksvm, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#KSVM rbfdot Method
CrossTable(data_test_ksvm_rbfdot$diagnosis,model_data_test_ksvm_rbfdot, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

#Log Reg Method
CrossTable(data_test_logr$diagnosis,predicted_diagnosis, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual','predicted'))

# Comparision of confusionMatrix


confusionMatrix(data_test_nb$diagnosis, model_nbayes_test, dnn = c(" NavieB Prediction", "Reference"))
confusionMatrix(data_test_km$diagnosis, data_test_km$clusterid, dnn = c(" KMeans Prediction", "Reference"))
confusionMatrix(data_knn_test_pred, data_test_knn_cl, dnn = c(" KNN Prediction", "Reference"))
confusionMatrix(data_test_rpart$diagnosis,model_data_test_rpart, dnn = c(" RpartDT Prediction", "Reference"))
confusionMatrix(data_test_ctree$diagnosis,model_data_test_ctree, dnn = c(" CtreeDT Prediction", "Reference"))
confusionMatrix(data_test_tree$diagnosis,model_data_test_tree, dnn = c(" TreeDT Prediction", "Reference"))
confusionMatrix(data_test_c5$diagnosis,model_data_test_c5, dnn = c(" C5Dt Prediction", "Reference"))
confusionMatrix(data_test_nn$diagnosis,predicted_diagnosis, dnn = c(" NN Prediction", "Reference"))
confusionMatrix(data_test_ksvm$diagnosis,model_data_test_ksvm, dnn = c(" KSVM Prediction", "Reference"))
confusionMatrix(data_test_ksvm_rbfdot$diagnosis,model_data_test_ksvm_rbfdot, dnn = c(" KSVM RB Prediction", "Reference"))
confusionMatrix(data_test_logr$diagnosis,predicted_diagnosis, dnn = c(" LogR Prediction", "Reference"))

