# 🚀 Big Data E-Commerce Analysis — Apache Spark + Scala

End-to-end Big Data pipeline analyzing **50,000 e-commerce transactions** using **Apache Spark + Scala**. Covers distributed data processing, advanced analytics, Spark SQL, and Machine Learning with MLlib.

![Scala](https://img.shields.io/badge/Scala-2.12-red?style=flat-square&logo=scala)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-3.5-orange?style=flat-square&logo=apachespark)
![MLlib](https://img.shields.io/badge/Spark_MLlib-RF_Model-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-50K_Records-green?style=flat-square)

---

## 📊 What This Project Covers

| Section | Analysis |
|---------|---------|
| Data Loading | Schema inference, partitioning, caching |
| Data Quality | Null checks, duplicate detection |
| Business Metrics | Revenue, profit, margin, order value |
| Monthly Trends | MoM growth using Window Functions |
| Category Analysis | Revenue, profit margin, avg rating |
| City Performance | City ranking using RANK() |
| Customer Segmentation | RFM-style (VIP/Premium/Regular/New) |
| Spark SQL | Running totals, CTEs, top-N per group |
| ML — MLlib | Random Forest Revenue Prediction (R²=0.95+) |
| Output | Parquet files saved to disk |

---

## 🏗️ Project Structure

```
1_spark_big_data_analysis/
│
├── src/main/scala/com/mubashir/ecommerce/
│   └── ECommerceAnalysis.scala   ← Main Scala + Spark code
│
├── spark_runner.py               ← PySpark alternative (easier setup)
├── build.sbt                     ← Scala build configuration
├── project/
│   └── plugins.sbt               ← SBT plugins
│
├── ecommerce_transactions.csv    ← 50,000 transaction dataset
└── README.md
```

---

## ⚙️ How to Run

### Option 1: PySpark (Easiest — No Scala needed)

```bash
# Install PySpark
pip install pyspark

# Run analysis
python spark_runner.py
```

### Option 2: Scala + SBT

```bash
# Install Java 11+: https://adoptium.net/
# Install SBT: https://www.scala-sbt.org/

# Compile and run
sbt run

# Or build a JAR and submit to Spark cluster
sbt assembly
spark-submit --class com.mubashir.ecommerce.ECommerceAnalysis \
             target/scala-2.12/BigDataECommerceAnalysis-assembly-1.0.jar
```

### Option 3: Databricks (Cloud — No setup needed)

1. Upload `ecommerce_transactions.csv` to Databricks DBFS
2. Create a new notebook
3. Copy-paste code from `spark_runner.py`
4. Run all cells

---

## 🧠 Key Spark Concepts Used

- **SparkSession** with local[*] multi-core processing
- **DataFrame API** — filter, groupBy, agg, join
- **Window Functions** — LAG, RANK, SUM OVER, ROWS UNBOUNDED
- **Spark SQL** — createOrReplaceTempView, complex SQL queries
- **Caching** — `.cache()` for repeated use optimization
- **Spark MLlib** — Pipeline, StringIndexer, VectorAssembler, RandomForestRegressor
- **Parquet** — columnar storage format for output

---

## 📈 Dataset Info

| Field | Detail |
|-------|--------|
| Records | 50,000 transactions |
| Period | 2021 — 2024 |
| Cities | 10 Pakistani cities |
| Categories | 8 product categories |
| Features | 21 columns |
| Total Revenue | ~$2.9 Billion |

---

## 🤖 ML Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor (50 trees) |
| R² Score | 0.95+ |
| Features | 11 (price, qty, discount, category, city, etc.) |

---

*Built by Muhammad Mubashir*
