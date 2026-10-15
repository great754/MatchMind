"""Run inside a Fabric Spark notebook after uploading the CSV snapshot.

Not executed locally: no Fabric workspace/session is configured.
"""
def import_snapshot(spark,files_root,manifest):
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType,StructField,StringType,LongType,DoubleType
    types={'string':StringType(),'int64':LongType(),'float64':DoubleType()}
    counts={}
    for name,spec in manifest['tables'].items():
        schema=StructType([StructField(c,types[spec['column_types'][c]],True) for c in spec['columns']])
        frame=spark.read.option('header',True).option('multiLine',True).option('quote','"').option('escape','"').schema(schema).csv(files_root+'/'+name+'.csv')
        count=frame.count()
        if count!=spec['rows']:raise ValueError('Unexpected row count: '+name)
        if frame.select(*spec['primary_key']).distinct().count()!=count:raise ValueError('Duplicate keys: '+name)
        for key in spec['primary_key']:
            if frame.filter(F.col(key).isNull()).limit(1).count():raise ValueError('Missing primary key: '+name)
        frame.write.format('delta').mode('errorifexists').saveAsTable('matchmind_'+name.lower())
        counts[name]=count
    return counts
