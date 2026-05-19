from pyspark import pipelines as dp
from pyspark.sql.functions import col, to_date, expr
from pyspark.sql.functions import sum as _sum, countDistinct, date_trunc

# Unity Catalog Volumes paths
LANDING = "/Volumes/dlt_demo/tpch/tpch_landing"
SCHEMA_LOC = "/Volumes/dlt_demo/tpch/tpch_schema"

# -----------------------
# BRONZE (Streaming tables using Auto Loader on landing files)
# -----------------------
@dp.table(
    name="bronze_orders",
    comment="Bronze orders ingested incrementally from Volume landing via Auto Loader"
)
def bronze_orders():
    return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{SCHEMA_LOC}/orders")
        .load(f"{LANDING}/orders")
    )

@dp.table(
    name="bronze_lineitem",
    comment="Bronze lineitem ingested incrementally from Volume landing via Auto Loader"
)
def bronze_lineitem():
    return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{SCHEMA_LOC}/lineitem")
        .load(f"{LANDING}/lineitem")
    )

@dp.table(
    name="bronze_customer",
    comment="Bronze customer ingested incrementally from Volume landing via Auto Loader"
)
def bronze_customer():
    return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{SCHEMA_LOC}/customer")
        .load(f"{LANDING}/customer")
    )

# -----------------------
# SILVER (Clean + Expectations)
# Expectations are part of the pipelines API and tracked in pipeline metrics.
# -----------------------
@dp.table(name="silver_orders", comment="Orders cleaned and typed with quality checks")
@dp.expect_or_fail("order_id_not_null", "order_id IS NOT NULL")
@dp.expect_or_drop("customer_id_not_null", "customer_id IS NOT NULL")
def silver_orders():
    df = dp.read_stream("bronze_orders")
    return (df.select(
            col("o_orderkey").alias("order_id"),
            col("o_custkey").alias("customer_id"),
            to_date(col("o_orderdate")).alias("order_date"),
            col("o_orderstatus"),
            col("o_totalprice").cast("decimal(18,2)").alias("total_price")
        ))

@dp.table(name="silver_lineitem", comment="Lineitem cleaned + revenue derived with quality rules")
@dp.expect_or_drop("quantity_positive", "quantity > 0")
@dp.expect_or_drop("extendedprice_non_negative", "extended_price >= 0")
def silver_lineitem():
    df = dp.read_stream("bronze_lineitem")
    return (df.select(
            col("l_orderkey").alias("order_id"),
            col("l_suppkey").alias("supplier_id"),
            col("l_partkey").alias("part_id"),
            col("l_quantity").cast("decimal(18,2)").alias("quantity"),
            col("l_extendedprice").cast("decimal(18,2)").alias("extended_price"),
            col("l_discount").cast("decimal(5,2)").alias("discount"),
            (col("l_extendedprice") * expr("1 - l_discount")).alias("revenue")
        ))

@dp.table(name="silver_customer", comment="Customer dimension cleaned with quality rules")
@dp.expect_or_fail("customer_id_not_null", "customer_id IS NOT NULL")
def silver_customer():
    df = dp.read_stream("bronze_customer")
    return (df.select(
            col("c_custkey").alias("customer_id"),
            col("c_name").alias("customer_name"),
            col("c_mktsegment").alias("market_segment"),
            col("c_acctbal").cast("decimal(18,2)").alias("account_balance")
        ))

# -----------------------
# GOLD (Materialized views)
# -----------------------
@dp.materialized_view(name="gold_customer_revenue", comment="Revenue and order counts per customer")
def gold_customer_revenue():
    o = dp.read("silver_orders")
    l = dp.read("silver_lineitem")
    c = dp.read("silver_customer")

    fact = o.join(l, "order_id").join(c, "customer_id")

    return (fact.groupBy("customer_id", "customer_name")
                .agg(
                    _sum("revenue").alias("total_revenue"),
                    countDistinct("order_id").alias("total_orders")
                ))

@dp.materialized_view(name="gold_monthly_revenue", comment="Monthly revenue trend")
def gold_monthly_revenue():
    o = dp.read("silver_orders")
    l = dp.read("silver_lineitem")
    fact = o.join(l, "order_id")

    return (fact.groupBy(date_trunc("month", col("order_date")).alias("month"))
                .agg(_sum("revenue").cast("decimal(18,2)").alias("monthly_revenue")))