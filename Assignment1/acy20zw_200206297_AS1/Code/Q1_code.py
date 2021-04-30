"""Q1_A"""

from pyspark.sql import SparkSession
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import numpy as np
from datetime import datetime
import seaborn as sns
from operator import add

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Q1_A") \
    .config("spark.local.dir","/fastdata/acy20zw") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile = spark.read.text("../../Data/NASA_access_log_Jul95.gz").cache()

print("Q1.A")
print("======================1======================")
hostsJapan = logFile.filter(logFile.value.contains(".ac.jp"))
num_hostsJapan = hostsJapan.count()
print(f"There are {num_hostsJapan} hosts form Japanese universities")

print("======================2======================")
hostsUK = logFile.filter(logFile.value.contains(".ac.uk"))
num_hostsUK = hostsUK.count()
print(f"There are {num_hostsUK} hosts form UK universities")

print("======================3======================")
def get_again(target):
    last = target.rsplit(".")[-1]
    return last=="edu"
hostsUS = logFile.filter(logFile.value.contains(".edu"))
data_US = hostsUS.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
num_hostsUS = data_US.select('host').rdd.map(lambda e: e.host).filter(lambda e: get_again(e)).count()
print(f"There are {num_hostsUS} hosts form US universities")

hosts = ('Japan', 'UK', 'US')
hosts_num = (num_hostsJapan, num_hostsUK, num_hostsUS)
plt.figure(figsize=(10,10))
plt.bar(hosts, hosts_num)
plt.title("Number of requests for Countries", fontsize=30)
plt.xlabel("Country")
plt.ylabel("Total Numbers")
plt.savefig("../Output/Q1_Aplot.png")

print("Q1_B")
print("======================1======================")
print("=====Japan=====")
hostsJP = logFile.filter(logFile.value.contains(".ac.jp"))
data_JP = hostsJP.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
hostsJP_count = data_JP.select('host').rdd.map(lambda e: e.host).map(lambda e: (e.rsplit(".")[-3], 1)).reduceByKey(lambda x,y:x+y).sortBy(lambda e:e[1], ascending=False).toDF(["University","count"])
hostsJP_count.show(9,False)

print("=====uk=====")
hostsUK = logFile.filter(logFile.value.contains(".ac.uk"))
data_UK = hostsUK.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
hostsUK_count = data_UK.select('host').rdd.map(lambda e: e.host).map(lambda e: (e.rsplit(".")[-3], 1)).reduceByKey(lambda x,y:x+y).sortBy(lambda e:e[1], ascending=False).toDF(["University","count"])
hostsUK_count.show(9,False)

print("=====US=====")
def get_again(target):
    last = target.rsplit(".")[-1]
    return last=="edu"

hostsUS_count = data_US.select('host').rdd.map(lambda e: e.host).filter(lambda e: get_again(e)).map(lambda e: (e.rsplit(".")[-2], 1)).reduceByKey(lambda x,y:x+y).sortBy(lambda e:e[1], ascending=False).toDF(["University","count"])
hostsUS_count.show(9,False)

#======================2======================
def prepare_pltData(data):
    count = data.select('count').collect()
    name = data.select('University').collect()
    # Get the hosts name to plot
    name_arr = [str(row['University']) for row in name]
    nameplt = name_arr[:9]
    nameplt.append("Others")
    # Get the count number to plot
    count_arr = [int(row['count']) for row in count]
    countplt = count_arr[:9]
    countplt.append(sum(count_arr[9:]))
    return nameplt, countplt
    
hostsJP_nameplt, hostsJP_countplt = prepare_pltData(hostsJP_count)

plt.figure(figsize=(10,10))
plt.pie(hostsJP_countplt, labels=hostsJP_nameplt, autopct='%1.1f%%', pctdistance=0.9)
plt.title("Pie chart of Japan", fontsize=30)
plt.savefig("../Output/Q1_BJapan.png")

#Pie char for UK
hostsUK_nameplt, hostsUK_countplt = prepare_pltData(hostsUK_count)

plt.figure(figsize=(10,10))
plt.pie(hostsUK_countplt, labels=hostsUK_nameplt, autopct='%1.1f%%', pctdistance=0.9)
plt.title("Pie chart of UK", fontsize=30)
plt.savefig("../Output/Q1_BUK.png")


# Pie char for US
hostsUS_nameplt, hostsUS_countplt = prepare_pltData(hostsUS_count)

plt.figure(figsize=(10,10))
plt.pie(hostsUS_countplt, labels=hostsUS_nameplt, autopct='%1.1f%%', pctdistance=0.9)
plt.title("Pie chart of US", fontsize=30)
plt.savefig("../Output/Q1_BUS.png")


#Q1_C
#============================================
#=====Japan=====
most_frequentJP = data_JP.select('host').rdd.map(lambda e: e.host).map(lambda e: (e.rsplit(".")[-3], 1)).reduceByKey(lambda x,y:x+y).sortBy(lambda e:e[1], ascending=False).toDF(["University","count"]).select("University").first().University

mostJP = logFile.filter(logFile.value.contains(most_frequentJP))
data_mostJP = mostJP.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

hostsJP_count = data_mostJP.select('timestamp').groupBy('timestamp').count().rdd.map(lambda e: e.timestamp).map(lambda e: e.rsplit(" ")[0])

hostsJP_time = hostsJP_count.map(lambda e: datetime.strptime(e, "%d/%b/%Y:%H:%M:%S")).map(lambda e: [e.day, e.hour]).toDF(["Day","Hour"]).groupBy(["Day","Hour"]).count()

x_JPmax = (1,32) if hostsJP_time.select("Day").sort("Day").rdd.max()["Day"] > 25 else (5,26)
xJP_label = [i for i in range(*x_JPmax)]
yJP_label = [i for i in range(0, 24)]

JP_pltMatrix = np.zeros((24, x_JPmax[1]-x_JPmax[0]))

for row in hostsJP_time.collect():
    x = row['Hour']
    y= row['Day'] - x_JPmax[0]
    val = row['count']
    JP_pltMatrix[x,y] = val

plt.figure(figsize=(10,10))
sns.heatmap(JP_pltMatrix, annot=True, cmap="GnBu", xticklabels=xJP_label, yticklabels=yJP_label, fmt="g")
plt.title("The Heatmap of Japan", fontsize=30)
plt.xlabel("Day")
plt.ylabel("Hour")

plt.savefig("../Output/Q1_CJP.png")

#=====UK=====
most_frequentUK = data_UK.select('host').rdd.map(lambda e: e.host).map(lambda e: (e.rsplit(".")[-3], 1)).reduceByKey(lambda x,y:x+y).sortBy(lambda e:e[1], ascending=False).toDF(["University","count"]).select("University").first().University                

mostUK = logFile.filter(logFile.value.contains(most_frequentUK))
data_mostUK = mostUK.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
hostsUK_count = data_mostUK.select('timestamp').groupBy('timestamp').count().rdd.map(lambda e: e.timestamp).map(lambda e: e.rsplit(" ")[0])

hostsUK_time = hostsUK_count.map(lambda e: datetime.strptime(e, "%d/%b/%Y:%H:%M:%S")).map(lambda e: [e.day, e.hour]).toDF(["Day","Hour"]).groupBy(["Day","Hour"]).count()

x_UKmax = (1,32) if hostsUK_time.select("Day").sort("Day").rdd.max()["Day"] > 25 else (5,26)
xUK_label = [i for i in range(*x_UKmax)]
yUK_label = [i for i in range(0, 24)]

UK_pltMatrix = np.zeros((24, x_UKmax[1]-x_UKmax[0]))

for row in hostsUK_time.collect():
    x = row['Hour']
    y= row['Day'] - x_UKmax[0]
    val = row['count']
    UK_pltMatrix[x,y] = val

plt.figure(figsize=(10,10))
sns.heatmap(UK_pltMatrix, annot=True, cmap="GnBu", xticklabels=xUK_label, yticklabels=yUK_label, fmt="g")
plt.title("The Heatmap of UK", fontsize=30)
plt.xlabel("Day")
plt.ylabel("Hour")

plt.savefig("../Output/Q1_CUK.png")

#=====US=====
most_frequentUS = data_US.select('host').rdd.map(lambda e: e.host).filter(lambda e: get_again(e)).map(lambda e: (e.rsplit(".")[-2], 1)).reduceByKey(lambda x,y:x+y).sortBy(lambda e:e[1], ascending=False).toDF(["University","count"]).select("University").first().University

mostUS = logFile.filter(logFile.value.contains(most_frequentUS))
data_mostUS = mostUS.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
hostsUS_count = data_mostUS.select('timestamp').groupBy('timestamp').count().rdd.map(lambda e: e.timestamp).map(lambda e: e.rsplit(" ")[0])

hostsUS_time = hostsUS_count.map(lambda e: datetime.strptime(e, "%d/%b/%Y:%H:%M:%S")).map(lambda e: [e.day, e.hour]).toDF(["Day","Hour"]).groupBy(["Day","Hour"]).count()

x_USmax = (1,32) if hostsUS_time.select("Day").sort("Day").rdd.max()["Day"] > 25 else (5,26)
xUS_label = [i for i in range(*x_USmax)]
yUS_label = [i for i in range(0, 24)]

US_pltMatrix = np.zeros((24, x_USmax[1]-x_USmax[0]))

for row in hostsUS_time.collect():
    x = row['Hour']
    y= row['Day'] - x_USmax[0]
    val = row['count']
    US_pltMatrix[x,y] = val

plt.figure(figsize=(10,10))
sns.heatmap(US_pltMatrix, annot=True, cmap="GnBu", xticklabels=xUS_label, yticklabels=yUS_label, fmt="g")
plt.title("The Heatmap of US", fontsize=30)
plt.xlabel("Day")
plt.ylabel("Hour")

plt.savefig("../Output/Q1_CUS.png")

spark.stop()