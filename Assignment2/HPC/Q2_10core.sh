#!/bin/bash
#$ -l h_rt=8:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -P rse-com6012
#$ -l rmem=50G #number of memery
#$ -o ../Output/Q2_10core.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M zwang219@shef.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 50g --executor-memory 8g --master local[10] --conf spark.driver.maxResultSize=4g ../Code/Q2.py