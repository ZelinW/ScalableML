from pyspark.sql import SparkSession 
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.recommendation import ALS 
from pyspark.ml.linalg import Vectors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Q2_A") \
    .config("spark.local.dir","/fastdata/acy20zw") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

ratings = spark.read.load('../../Data/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true")
host_count = ratings.sort('timestamp', ascending=True)
total_count = ratings.count()
print("=============A=============")
def generate_data(orgin, percentage):
    orgin_with_index = orgin.rdd.zipWithIndex().cache()
    training_index = int(total_count * percentage)-1
    train = orgin_with_index.filter(lambda e : e[1]< training_index).map(lambda e: e[0]).toDF(["userId","movieId","rating","timestamp"])
    test = orgin_with_index.filter(lambda e : e[1] >= training_index).map(lambda e: e[0]).toDF(["userId","movieId","rating","timestamp"])
    return train, test

# 3 different splits of data
prepared_50_train, prepared_50_test = generate_data(host_count, 0.5)
prepared_65_train, prepared_65_test = generate_data(host_count, 0.65)
prepared_80_train, prepared_80_test = generate_data(host_count, 0.8)

def use_ALS(trainset, testset, als):
    model = als.fit(trainset)
    predictions = model.transform(testset)
    #Define evaluator
    eval_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
    eval_mse = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
    eval_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
    #Calculate
    rmse = eval_rmse.evaluate(predictions)
    mse = eval_mse.evaluate(predictions)
    mae = eval_mae.evaluate(predictions)
    return rmse, mse, mae, model
    

# ALS setting used in Lab3 with seed 200206297
myseed = 200206297
als_default = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
ALS1_rmse_50, ALS1_mse_50, ALS1_mar_50, model1_50 = use_ALS(prepared_50_train, prepared_50_test, als_default)
ALS1_rmse_65, ALS1_mse_65, ALS1_mar_65, model1_65 = use_ALS(prepared_65_train, prepared_65_test, als_default)
ALS1_rmse_80, ALS1_mse_80, ALS1_mar_80, model1_80 = use_ALS(prepared_80_train, prepared_80_test, als_default)
print("The defualt ALS")
asl1_result_df = sc.parallelize([(0.5, ALS1_rmse_50, ALS1_mse_50, ALS1_mar_50 ),(0.65, ALS1_rmse_65, ALS1_mse_65, ALS1_mar_65),(0.8, ALS1_rmse_80, ALS1_mse_80, ALS1_mar_80)]).toDF(["Train Split","RMSE","MSE","MAR"]).show()


# ALS setting with maxIter=15
als_2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop", maxIter=15)

ALS2_rmse_50, ALS2_mse_50, ALS2_mar_50, model2_50 = use_ALS(prepared_50_train, prepared_50_test, als_2)
ALS2_rmse_65, ALS2_mse_65, ALS2_mar_65, model2_65 = use_ALS(prepared_65_train, prepared_65_test, als_2)
ALS2_rmse_80, ALS2_mse_80, ALS2_mar_80, model2_80 = use_ALS(prepared_80_train, prepared_80_test, als_2)
print("ALS with maxIter=10")
asl2_result_df = sc.parallelize([(0.5, ALS2_rmse_50, ALS2_mse_50, ALS2_mar_50 ),(0.65, ALS2_rmse_65, ALS2_mse_65, ALS2_mar_65),(0.8, ALS2_rmse_80, ALS2_mse_80, ALS2_mar_80)]).toDF(["Train Split","RMSE","MSE","MAR"]).show()

# ALS setting with
als_3 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop", regParam = 0.1)

ALS3_rmse_50, ALS3_mse_50, ALS3_mar_50, model3_50 = use_ALS(prepared_50_train, prepared_50_test, als_3)
ALS3_rmse_65, ALS3_mse_65, ALS3_mar_65, model3_65 = use_ALS(prepared_65_train, prepared_65_test, als_3)
ALS3_rmse_80, ALS3_mse_80, ALS3_mar_80, model3_80 = use_ALS(prepared_80_train, prepared_80_test, als_3)
print("ALS with regParam=0.1")
asl3_result_df = sc.parallelize([(0.5, ALS3_rmse_50, ALS3_mse_50, ALS3_mar_50),(0.65, ALS2_rmse_65, ALS2_mse_65, ALS2_mar_65),(0.8, ALS2_rmse_80, ALS2_mse_80, ALS2_mar_80)]).toDF(["Train Split","RMSE","MSE","MAR"]).show()
#The table of ALS


###B.k_means
print("=============B=============")
print("====1====")
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

#User factors for each split
userFactors_50 = model1_50.userFactors
userFactors_65 = model1_65.userFactors
userFactors_80 = model1_80.userFactors

def use_kmeans(userFactors):
    kmeans = KMeans(k=20, seed=myseed)
    model = kmeans.fit(userFactors)
    transformed = model.transform(userFactors)
    k_top3 = transformed.select('prediction').groupBy('prediction').count().sort('count', ascending=False).head(3)
    first_cluster = k_top3[0]['prediction']
    first_count = k_top3[0]['count']
    second_cluster = k_top3[1]['prediction']
    second_count = k_top3[1]['count']
    third_cluster = k_top3[2]['prediction']
    third_count = k_top3[2]['count']
    table_data = [[first_cluster, first_count],[second_cluster, second_count],[third_cluster, third_count]]
    data_df = sc.parallelize(table_data).toDF(['clusters','count'])
    return data_df, first_cluster, transformed
    
data_df50, large_cluster50, transformed50 = use_kmeans(userFactors_50.select("features"))
print("Top3 clusters of trainset with 0.5")
data_df50.show(3,False)

data_df65, large_cluster65, transformed65 = use_kmeans(userFactors_65.select("features"))
print("Top3 clusters of trainset with 0.65")
data_df65.show(3,False)

data_df80, large_cluster80, transformed80 = use_kmeans(userFactors_80.select("features"))
print("Top3 clusters of trainset with 0.80")
data_df80.show(3,False)

print("====2====")
movies = spark.read.load('../../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true")

def get_userId(cluster_results, largest_cluster, userFactors):
    large_clu_set = cluster_results.where(cluster_results.prediction==largest_cluster)
    userId = large_clu_set.join(userFactors, "features", "inner").select("id").collect()
    userId_list = [row["id"] for row in userId]
    return userId_list

def get_movieId(data, userId_list):
    data = data.filter(data["rating"]>=4)
    movieId = data.rdd.filter(lambda e: e.userId in userId_list).map(lambda e: e.movieId).distinct().collect()
    return movieId

def get_genres(movie, movieId_list):
    genres = movie.rdd.filter(lambda e: e.movieId in movieId_list).map(lambda e: e.genres.split("|")).flatMap(lambda e: e).map(lambda e: (e, )).toDF(["genres"]).groupBy("genres").count().sort("count", ascending = False)
    return genres
    
def whole(transformed, large_cluster, userFactors, target_train_data, target_test_data, movies):
    userId = get_userId(transformed, large_cluster, userFactors)
    movieId_train = get_movieId(target_train_data, userId)
    genres_train = get_genres(movies, movieId_train)
    movieId_test = get_movieId(target_test_data, userId)
    genres_test = get_genres(movies, movieId_test)
    return genres_train, genres_test


print("===For the first split with 0.5===")
genres_train50, genres_test50 = whole(transformed50, large_cluster50, userFactors_50, prepared_50_train, prepared_50_test, movies)
print("trainset")
genres_train50.show(5)
print("testset")
genres_test50.show(5)

print("For the second split with 0.65")
genres_train65, genres_test65 = whole(transformed65, large_cluster65, userFactors_65, prepared_65_train, prepared_65_test, movies)
print("trainset")
genres_train65.show(5)
print("testset")
genres_test65.show(5)

print("For the second split with 0.8")
genres_train80, genres_test80 = whole(transformed80, large_cluster80, userFactors_80, prepared_80_train, prepared_80_test, movies)
print("trainset")
genres_train80.show(5)
print("testset")
genres_test80.show(5)

spark.stop()