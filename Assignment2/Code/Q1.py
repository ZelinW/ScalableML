from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time

myseed = 200206297

spark = SparkSession.builder \
    .master("local[10]") \
    .appName("Q1") \
    .config("spark.local.dir","/fastdata/acy20zw") \
    .getOrCreate()
    
sc = spark.sparkContext
sc.setLogLevel("WARN")

#Load data
logFile = spark.read.csv("../../Data/HIGGS.csv.gz")
ncolumns = len(logFile.columns)
logFile = logFile.withColumnRenamed(logFile.columns[0], 'labels')

#Change variables to the type double
StringColumns = [x.name for x in logFile.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    logFile = logFile.withColumn(c, col(c).cast("double"))
#=====================================Split and store trainset and testset in whole dataset=================
# Split the dataset to trainset and testset only one time
data_training, data_test = logFile.randomSplit([0.8, 0.2], myseed)
# Store trainset and testset to disk

data_training.write.mode('overwrite').parquet('../../Data/HIGGS_training.parquet')
data_test.write.mode('overwrite').parquet('../../Data/HIGGS_test.parquet')

# Read data

data_training = spark.read.parquet('../../Data/HIGGS_training.parquet')
data_test = spark.read.parquet('../../Data/HIGGS_test.parquet')

data_training = data_training.cache()
data_test = data_test.cache()
#======================================Choose 1% of the whole dataset to built model=======================
demo, other = logFile.randomSplit([0.01, 0.99], myseed)
# Get trainset and testset from demo
demo_training, demo_test = demo.randomSplit([0.8, 0.2], myseed)
# Store the demo data and read it

demo_training.write.mode('overwrite').parquet('../../Data/demo_training.parquet')
demo_test.write.mode('overwrite').parquet('../../Data/demo_test.parquet')
# Read data
demo_training = spark.read.parquet('../../Data/demo_training.parquet')
demo_test = spark.read.parquet('../../Data/demo_test.parquet')

demo_training = demo_training.cache()
demo_test = demo_test.cache()

# Vectorise data
vecAssembler = VectorAssembler(inputCols = logFile.columns[1:], outputCol = 'features') 

evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
evaluator_auc = BinaryClassificationEvaluator\
      (labelCol="labels", rawPredictionCol="prediction", metricName="areaUnderROC")
#===============================================Part 1==============================================================
#=======================================Random Forest Classifier=====================================
# Built random forest classifier
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol='labels', featuresCol='features', maxDepth=10, numTrees=10, \
                            maxBins=5, impurity='entropy', seed=myseed)

# Combine stages into pipeline
stages = [vecAssembler, rf]
pipelineRFC = Pipeline(stages=stages)

# Create ParamGridBuilder
paramGridRFC = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxBins, [5, 8, 10]) \
    .build()

# Make Crossvalidator object
crossvalRFC = CrossValidator(estimator=pipelineRFC,
                          estimatorParamMaps=paramGridRFC,
                          evaluator=evaluator,
                          numFolds=3)
                  
#Find the best parameter        
cvModelRFC = crossvalRFC.fit(demo_training)
predictionRFC = cvModelRFC.transform(demo_test)
accuracyRFC = evaluator.evaluate(predictionRFC)
auc_RFC = evaluator_auc.evaluate(predictionRFC)

print("Accuracy for best RFC model = %g " % accuracyRFC)
print("AUC for best RFC model = %g " % auc_RFC)

paramDictRFC = {param[0].name: param[1] for param in cvModelRFC.bestModel.stages[-1].extractParamMap().items()}
best_RFC_maxDepth = paramDictRFC['maxDepth']
best_RFC_numTrees = paramDictRFC['numTrees']
best_RFC_maxBins = paramDictRFC['maxBins']

print("The best parameters for RF is: maxDepth:{}, numTrees:{}, maxBins{}.".format(best_RFC_maxDepth, best_RFC_numTrees, best_RFC_maxBins))
#=======================================Gradient boosting Classifier==================================
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol='labels', featuresCol='features', maxIter=5, maxDepth=2, maxBins=5)

# Combine stages into pipeline
stagesGBT = [vecAssembler, gbt]
pipelineGBT = Pipeline(stages=stagesGBT)

# Create ParamGridBuilder
paramGridGBT = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [5, 10, 15]) \
    .addGrid(gbt.maxDepth, [2, 4, 8]) \
    .addGrid(gbt.maxBins, [5, 8, 10]) \
    .build()

# Make Crossvalidator object
crossvalGBT = CrossValidator(estimator=pipelineGBT,
                          estimatorParamMaps=paramGridGBT,
                          evaluator=evaluator,
                          numFolds=3)
                  
#Find the best parameter        
cvModelGBT = crossvalGBT.fit(demo_training)
predictionGBT = cvModelGBT.transform(demo_test)
accuracyGBT = evaluator.evaluate(predictionGBT)
auc_GBT = evaluator_auc.evaluate(predictionGBT)

print("Accuracy for best GBT model = %g " % accuracyGBT)
print("AUC for best GBT model = %g " % auc_GBT)

paramDictGBT = {param[0].name: param[1] for param in cvModelGBT.bestModel.stages[-1].extractParamMap().items()}
best_GBT_maxIter = paramDictGBT['maxIter']
best_GBT_maxDepth = paramDictGBT['maxDepth']
best_GBT_maxBins = paramDictGBT['maxBins']

print("The best parameters for GBT is: maxIter:{}, numDepth:{}, maxBins{}.".format(best_GBT_maxIter, best_GBT_maxDepth, best_GBT_maxBins))
#=======================================Neural Networks Classifier====================================
from pyspark.ml.classification import MultilayerPerceptronClassifier
mpc = MultilayerPerceptronClassifier( labelCol="labels", featuresCol="features",maxIter=20, layers=[28,2],\
                                     stepSize=0.03, seed=myseed)

# Combine stages into pipeline
stagesMPC = [vecAssembler, mpc]
pipelineMPC = Pipeline(stages=stagesMPC)

# Create ParamGridBuilder
paramGridMPC = ParamGridBuilder() \
    .addGrid(mpc.maxIter, [10, 20, 30]) \
    .addGrid(mpc.layers, [[28, 2], [28, 100, 2], [28, 200, 2], [28, 100, 200, 2]]) \
    .addGrid(mpc.stepSize, [0.03, 0.07, 0.1]) \
    .build()

# Make Crossvalidator object
crossvalMPC = CrossValidator(estimator=pipelineMPC,
                          estimatorParamMaps=paramGridMPC,
                          evaluator=evaluator,
                          numFolds=3)
                  
#Find the best parameter        
cvModelMPC = crossvalMPC.fit(demo_training)
predictionMPC = cvModelMPC.transform(demo_test)
accuracyMPC = evaluator.evaluate(predictionMPC)
auc_MPC = evaluator_auc.evaluate(predictionMPC)

print("Accuracy for best MPC model = %g " % accuracyMPC)
print("AUC for best MPC model = %g " % auc_MPC)

paramDictMPC = {param[0].name: param[1] for param in cvModelMPC.bestModel.stages[-1].extractParamMap().items()}
best_MPC_maxIter = paramDictMPC['maxIter']
best_MPC_layers = paramDictMPC['layers']
best_MPC_stepSize = paramDictMPC['stepSize']

print("The best parameters for MPC is: maxIter:{}, layers:{}, stepSize{}.".format(best_MPC_maxIter, best_MPC_layers, best_MPC_stepSize))

#======================================================Part 2=======================================================
data_training = data_training.dropna(how='any')
data_training = vecAssembler.transform(data_training)

data_test = data_test.dropna(how='any')
data_test = vecAssembler.transform(data_test)
#==================RFC===================
RFC_start = time.time()
rf = RandomForestClassifier(labelCol='labels', featuresCol='features', maxDepth=best_RFC_maxDepth, numTrees=best_RFC_numTrees, \
                            maxBins=best_RFC_maxBins, impurity='entropy', seed=myseed)
                            
RFC_model = rf.fit(data_training)
pred_RFC_all = RFC_model.transform(data_test)
accuracy_RFC_all = evaluator.evaluate(pred_RFC_all)
auc_RFC_all = evaluator_auc.evaluate(pred_RFC_all)

RFC_end = time.time()

import pandas as pd
featureImpRF = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), RFC_model.featureImportances)),
  columns=["feature", "importance"])
featureImpRF = featureImpRF.sort_values(by="importance", ascending=False)

RFC_time = RFC_end-RFC_start
print("Gradient boosting Classifier:{} s".format(RFC_time))
print("The accuracy of RFC with the larger dataset= %g " % accuracy_RFC_all)
print("The AUC of RFC is %g" % auc_RFC_all)
print("The most import feature is:", featureImpRF)
#================Gradient boosting Classifier====================
GBC_start = time.time()
gbc = GBTClassifier(labelCol='labels', featuresCol='features', maxIter=best_GBT_maxIter, maxDepth=best_GBT_maxDepth, maxBins=best_GBT_maxBins)

GBC_model = gbc.fit(data_training)
pred_GBC_all = GBC_model.transform(data_test)
accuracy_GBC_all = evaluator.evaluate(pred_GBC_all)
auc_GBC_all = evaluator_auc.evaluate(pred_GBC_all)

GBC_end = time.time()

featureImpGBC = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), GBC_model.featureImportances)),
  columns=["feature", "importance"])
featureImpGBC.sort_values(by="importance", ascending=False)

GBC_time = GBC_end-GBC_start
print("Gradient boosting Classifier:{} s".format(GBC_time))
print("The accuracy of GBC with the larger dataset= %g " % accuracy_GBC_all)
print("The AUC of GBC is %g" % auc_GBC_all)
print("The most import feature is:", featureImpGBC)
#================Neural networks Classifier=====================
MPC_start = time.time()

mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features",maxIter=best_MPC_maxIter, layers=best_MPC_layers,\
                                     stepSize=best_MPC_stepSize, seed=myseed)

MPC_model = mpc.fit(data_training)
pred_MPC_all = MPC_model.transform(data_test)
accuracy_MPC_all = evaluator.evaluate(pred_MPC_all)
auc_MPC_all = evaluator_auc.evaluate(pred_MPC_all)

MPC_end = time.time()

MPC_time = MPC_end-MPC_start
print("Neural networks classifier spent:{} s".format(MPC_time))
print("The accuracy of neural networks classifier with the larger dataset= %g " % accuracy_MPC_all)
print("The AUC of MPC is %g" % auc_MPC_all)
print("The total time of runing the three models is {}".format(RFC_time + GBC_time + MPC_time))

spark.stop()