from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import col, isnull
import time

myseed= 200206297

spark = SparkSession.builder \
    .master("local[10]") \
    .appName("Q1") \
    .config("spark.local.dir","/fastdata/acy20zw") \
    .getOrCreate()
    
sc = spark.sparkContext
sc.setLogLevel("WARN")


logFile = spark.read.csv('../../Data/train_set.csv', header=True)
columns_name = logFile.columns

prepare_file = logFile.replace('?', None).dropna(thresh=33)
# Replace Null to the mode of each row

mode_dic = {}
for col in columns_name:
    d = prepare_file.select(col).groupBy(col).count()
    if d.first()[0] == None:
        mode = d.limit(2).collect()[1][col]
    else:
        mode = d.first()[0]
    mode_dic[col] = mode
prepare_file = prepare_file.fillna(mode_dic)


# Drop some columns which is similar or useless to reduce the amount of calculation
need_drop = ['Row_ID', 'Household_ID', 'Var1', 'Var4', 'Var6', 'Blind_Make',
             'Blind_Model', 'Blind_Submodel','Cat2', 'Cat4', 'Cat5', 'Cat7', 'OrdCat']
for col in need_drop:
    prepare_file = prepare_file.drop(col)

columns_name = prepare_file.columns

# Convert all data to type double to find which column's elements can not be convert to double
# Then select those and convert them to index by StringIndexer
from pyspark.sql.functions import col, isnull

file_double = prepare_file
for c in columns_name:
    file_double = file_double.withColumn(c, col(c).cast("double"))

# Choose those columns which can not be converted to double
from pyspark.sql.functions import col, isnull
character_col_list = list()
for col in columns_name:
    if file_double.select(isnull(col)).first()[0]:
        character_col_list.append(col)

# Convert other columns to double
from pyspark.sql.functions import col, isnull
num_col_list = [x for x in columns_name if x not in character_col_list]
for c in num_col_list:
    prepare_file = prepare_file.withColumn(c, col(c).cast("double"))

# Use StringIndexer to convert those columns which can not be converted to double to index
from pyspark.ml.feature import StringIndexer
i = 0
for col in character_col_list:
    stringIndexer = StringIndexer(inputCol=col, outputCol='feature{}'.format(i))
    model = stringIndexer.fit(prepare_file)
    prepare_file = model.transform(prepare_file)
    i += 1

for col in character_col_list:
    prepare_file = prepare_file.drop(col)

# Vectorise    
from pyspark.ml.feature import VectorAssembler
features_data = prepare_file.drop("Claim_Amount")
vecAssembler = VectorAssembler(inputCols = features_data.columns, outputCol = 'features') 
dataVectorised = vecAssembler.transform(prepare_file).select('features', 'Claim_Amount')

# Downsample
data_Claim0 = dataVectorised[dataVectorised['Claim_Amount'] == 0]
data_Claim_else = dataVectorised[dataVectorised['Claim_Amount'] != 0]

dataVectorised = data_Claim_else.union(data_Claim0.sample(data_Claim_else.count()/data_Claim0.count()))

# Split data
training_set, test_set = dataVectorised.randomSplit([0.7, 0.3], seed=myseed)

from pyspark.ml.evaluation import RegressionEvaluator
evaluatorMSE = RegressionEvaluator(predictionCol='prediction', labelCol='Claim_Amount', metricName='mse')
evaluatorMAE = RegressionEvaluator(predictionCol='prediction', labelCol='Claim_Amount', metricName='mae')

# Linear Regression
from pyspark.ml.regression import LinearRegression
LR_start = time.time()
lr = LinearRegression(featuresCol='features', labelCol='Claim_Amount', regParam=0.06, elasticNetParam=0, maxIter=10)
model = lr.fit(training_set)
predict = model.transform(test_set)
LR_end = time.time()
LR_time = LR_end-LR_start

mse = evaluatorMSE.evaluate(predict)
mae = evaluatorMAE.evaluate(predict)

print('The mse of the model:{}'.format(mse))
print('The mae of the model:{}'.format(mae))
print('Time of LR model:{}'.format(LR_time))

# Conver the label of data which has non-zero label to 1

from pyspark.sql.functions import when
train_set1 = training_set.withColumn('Claim_Amount',when(dataVectorised.Claim_Amount!=0, 1).otherwise(0))
test_set1 = test_set.withColumn('Claim_Amount',when(dataVectorised.Claim_Amount!=0, 1).otherwise(0))

# The binary classifier
model_start = time.time()
from pyspark.ml.classification import RandomForestClassifier
rfc = RandomForestClassifier(featuresCol='features', labelCol='Claim_Amount', maxDepth=5, numTrees=3, seed=myseed)
RFC_model = rfc.fit(train_set1)

# Gamma Regressor
from pyspark.ml.regression import GeneralizedLinearRegression
glr = GeneralizedLinearRegression(featuresCol='features', labelCol='Claim_Amount', link='identity')
GLR_model = glr.fit(data_Claim_else)

# Combine the two model
predict_RFC = RFC_model.transform(test_set)
# select the results which predicted as 1
RFC_result = predict_RFC[predict_RFC['prediction']==1].select('features','Claim_Amount')
GLR_result = GLR_model.transform(RFC_result)
model_end = time.time()

mse = evaluatorMSE.evaluate(GLR_result)
mae = evaluatorMAE.evaluate(GLR_result)
print('mse :', mse)
print('mae :', mae)
print('Time:', model_end-model_start)

spark.stop()