import dlt

@dlt.table
def tansformation():
    return spark.range(10)