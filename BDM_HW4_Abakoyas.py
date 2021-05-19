Python 3.8.1 (v3.8.1:1b293b6006, Dec 18 2019, 14:08:53) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import statistics
import datetime
import json
import csv
import numpy as np
import sys
 
def main(sc, spark):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
    dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
    OUTPUT_PREFIX = sys.argv[1]

    CAT_CODES = {'445210', '722515', '445299', '445120', '452210', '311811', '722410', '722511', '445220', '445292', '445110', '445291', '445230', '446191', '446110', '722513', '452311'}
    CAT_GROUP = {'452311': 0, '452210': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446191': 5, '446110': 5, '722515': 6, '311811': 6, '445299': 7, '445220': 7, '445292': 7, '445291': 7, '445230': 7, '445210': 7, '445110': 8}

    dfD = dfPlaces.select("placekey","naics_code").where(dfPlaces["naics_code"].isin(CAT_CODES))
"""dfD = dfPlaces.select("placekey","naics_code")
dfD.where(F.col("placekey").isin(CAT_CODES)).show()
"""


    udfToGroup = F.udf(lambda x: CAT_GROUP[x], T.IntegerType())

    dfE = dfD.withColumn('group', udfToGroup('naics_code'))

    dfF = dfE.drop('naics_code').cache()

    groupCount = dfF.groupBy('group')\
            .agg(F.count('placekey'))\
            .toPandas()\
            .set_index('group')['count(placekey)'].to_dict()

    def expandVisits(date_range_start, visits_by_day):
       start = datetime.datetime(*map(int, date_range_start[:10].split('-')))
       return [((start + datetime.timedelta(days=days)).year, str((start + datetime.timedelta(days=days)))[5:10],visits) 
           for days,visits in enumerate(json.loads(visits_by_day))]

    visitType = T.StructType([T.StructField('year', T.IntegerType()),
                          T.StructField('date', T.StringType()),
                          T.StructField('visits', T.IntegerType())])

    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

    dfH = dfPattern.join(dfF, 'placekey') \
       .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
      .select('group', 'expanded.*')

    def computeStats(group, visits):
      if len(visits)<groupCount[group]:
        visits+=[0]*(groupCount[group]-len(visits))
      std = statistics.stdev(visits)+0.5
      med = statistics.median(visits)+0.5
      high = int(med+std)
      if med <=  std :
        low = 0
      else:
        low = int(med-std)
      return (int(med),low,high)

    statsType = T.StructType([T.StructField('median', T.IntegerType()),
                          T.StructField('low', T.IntegerType()),
                          T.StructField('high', T.IntegerType())])

    udfComputeStats = F.udf(computeStats, statsType)

    dfI = dfH.groupBy('group', 'year', 'date') \
      .agg(F.collect_list('visits').alias('visits')) \
      .withColumn('stats', udfComputeStats('group', 'visits'))


    def Date(date):
       return '2020-'+date

    udfDate = F.udf(Date,T.StringType())

    dfJ = dfI \
     .select('group','year',
        udfDate('date').alias('date'),
        'stats.*')\
        .sort(['group','year','date'])\
    .cache()

    csv_names=['Big Box Grocers',
'Convenience Stores',
'Drinking Places',
'Full-Service Restaurants',
'Limited-Service Restaurants',
'Pharmacies and Drug Stores',
'Snack and Bakeries',
'Specialty Food Stores',
'Supermarkets (except Convenience Stores)']
    toFileName = lambda x:'_'.join((''.join(map(lambda c: c if c.isalnum() else ' ', x.lower()))).split())
    for i,filename in enumerate(map(toFileName, csv_names)):
     dfJ.filter(dfJ.group==i) \
     .drop('group') \
     .coalesce(1) \
     .write.csv(f'{OUTPUT_PREFIX}/{filename}',
              mode='overwrite', header=True)







if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)