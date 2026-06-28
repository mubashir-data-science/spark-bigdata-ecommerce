"""
🚀 Big Data E-Commerce Analysis — PySpark Runner
Author: Muhammad Mubashir

Alternative: Run this if you don't have Scala/SBT installed.
Uses PySpark (Python API for Apache Spark) — same logic as Scala version.

Install: pip install pyspark
Run    : python spark_runner.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# ── 0. SparkSession ────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ECommerce Big Data Analysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n" + "="*60)
print("  🚀 BIG DATA E-COMMERCE ANALYSIS — Muhammad Mubashir")
print("="*60 + "\n")

# ── 1. LOAD DATA ───────────────────────────────────────────────
print("📂 SECTION 1: Loading Data...")
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("ecommerce_transactions.csv") \
    .cache()

total_rows = df.count()
print(f"✅ Rows    : {total_rows:,}")
print(f"   Columns : {len(df.columns)}")
print(f"   Parts   : {df.rdd.getNumPartitions}")
df.printSchema()
df.show(5)

# ── 2. OVERALL METRICS ─────────────────────────────────────────
print("\n💰 SECTION 2: Business Metrics...")
df.agg(
    count("transaction_id").alias("total_transactions"),
    countDistinct("customer_id").alias("unique_customers"),
    round(sum("revenue"), 2).alias("total_revenue"),
    round(sum("profit"),  2).alias("total_profit"),
    round(avg("revenue"), 2).alias("avg_order_value"),
    round(sum("profit") / sum("revenue") * 100, 2).alias("profit_margin_pct"),
    round(avg("rating"),  2).alias("avg_rating")
).show(truncate=False)

# ── 3. MONTHLY TREND ───────────────────────────────────────────
print("\n📅 SECTION 3: Monthly Revenue Trend...")
window_spec = Window.orderBy("year","month")
monthly = df.groupBy("year","month").agg(
    count("transaction_id").alias("orders"),
    round(sum("revenue"), 2).alias("revenue"),
    round(sum("profit"),  2).alias("profit"),
    round(avg("revenue"), 2).alias("avg_order_value")
).orderBy("year","month") \
 .withColumn("prev_revenue", lag("revenue", 1).over(window_spec)) \
 .withColumn("mom_growth_pct",
     round((col("revenue") - col("prev_revenue")) / col("prev_revenue") * 100, 2))

monthly.show(24, truncate=False)

# ── 4. CATEGORY PERFORMANCE ────────────────────────────────────
print("\n🏷️  SECTION 4: Category Performance...")
df.groupBy("category").agg(
    count("transaction_id").alias("transactions"),
    sum("quantity").alias("units_sold"),
    round(sum("revenue"), 2).alias("revenue"),
    round(sum("profit"),  2).alias("profit"),
    round(sum("profit") / sum("revenue") * 100, 2).alias("margin_pct"),
    round(avg("rating"),  2).alias("avg_rating")
).orderBy(desc("revenue")).show(truncate=False)

# ── 5. CITY PERFORMANCE ────────────────────────────────────────
print("\n🏙️  SECTION 5: City Performance...")
city_window = Window.orderBy(desc("revenue"))
df.groupBy("city").agg(
    count("transaction_id").alias("orders"),
    countDistinct("customer_id").alias("customers"),
    round(sum("revenue"), 2).alias("revenue"),
    round(avg("revenue"), 2).alias("avg_order_value")
).withColumn("rank", rank().over(city_window)) \
 .orderBy("rank").show(truncate=False)

# ── 6. SPARK SQL QUERIES ───────────────────────────────────────
print("\n🔧 SECTION 6: Spark SQL Analytics...")
df.createOrReplaceTempView("transactions")

# Running cumulative revenue
print("Running Total Revenue:")
spark.sql("""
    SELECT year, month,
           ROUND(SUM(revenue), 2) AS monthly_revenue,
           ROUND(SUM(SUM(revenue)) OVER (ORDER BY year, month
                 ROWS UNBOUNDED PRECEDING), 2) AS cumulative_revenue
    FROM transactions
    GROUP BY year, month ORDER BY year, month
""").show(24, truncate=False)

# Top product per category
print("Top Product per Category:")
spark.sql("""
    SELECT category, product, revenue, rank
    FROM (
        SELECT category, product,
               ROUND(SUM(revenue), 2) AS revenue,
               RANK() OVER (PARTITION BY category ORDER BY SUM(revenue) DESC) AS rank
        FROM transactions GROUP BY category, product
    ) WHERE rank = 1 ORDER BY revenue DESC
""").show(truncate=False)

# Day of week
print("Sales by Day of Week:")
spark.sql("""
    SELECT day_of_week, COUNT(*) AS orders,
           ROUND(SUM(revenue), 2) AS revenue,
           ROUND(AVG(revenue), 2) AS avg_value
    FROM transactions
    GROUP BY day_of_week ORDER BY revenue DESC
""").show(truncate=False)

# Return rate
print("Return Rate by Category:")
spark.sql("""
    SELECT category, COUNT(*) AS orders,
           SUM(is_returned) AS returned,
           ROUND(SUM(is_returned)*100.0/COUNT(*), 2) AS return_rate_pct
    FROM transactions
    GROUP BY category ORDER BY return_rate_pct DESC
""").show(truncate=False)

# ── 7. ML — REVENUE PREDICTION ────────────────────────────────
print("\n🤖 SECTION 7: ML Revenue Prediction (Spark MLlib)...")
indexers = [
    StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
    for c in ["customer_gender","category","city","payment_method"]
]
feature_cols = ["unit_price","quantity","discount","customer_age",
                "customer_gender_idx","category_idx","city_idx","payment_method_idx",
                "year","month","quarter"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestRegressor(labelCol="revenue", featuresCol="features",
                           numTrees=50, maxDepth=8, seed=42)
pipeline = Pipeline(stages=indexers + [assembler, rf])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count():,} | Test: {test_df.count():,}")

print("Training model...")
model = pipeline.fit(train_df)
preds = model.transform(test_df)

evaluator = RegressionEvaluator(labelCol="revenue", predictionCol="prediction")
rmse = evaluator.setMetricName("rmse").evaluate(preds)
r2   = evaluator.setMetricName("r2").evaluate(preds)
mae  = evaluator.setMetricName("mae").evaluate(preds)

print("="*45)
print("   ML MODEL PERFORMANCE")
print("="*45)
print(f"  RMSE : ${rmse:,.2f}")
print(f"  MAE  : ${mae:,.2f}")
print(f"  R²   : {r2:.4f}")
print("="*45)

preds.select("transaction_id","category","unit_price","quantity","revenue",
             round("prediction", 2).alias("predicted_revenue")) \
     .show(10, truncate=False)

# ── 8. SAVE RESULTS ───────────────────────────────────────────
print("\n💾 SECTION 8: Saving Parquet Results...")
monthly.write.mode("overwrite").parquet("output/monthly_revenue")
print("✅ Saved to output/monthly_revenue/")

spark.stop()
print("\n✅ BIG DATA ANALYSIS COMPLETE!")
