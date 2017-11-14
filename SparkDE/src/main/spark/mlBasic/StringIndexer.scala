package mlBasic

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object StringIndexer {
    def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .appName("StringIndexerSample")
      .master("local[*]")
      .getOrCreate()

    val df1 = spark.createDataFrame(Seq(
      (0, "red"),
      (1, "blue"),
      (2, "green"),
      (3, "red"))).toDF("id", "color")

    val strignIndexer = new StringIndexer()
      .setInputCol("color")
      .setOutputCol("colorIndex")
      .fit(df1)

    val df2 = strignIndexer.transform(df1)

    df2.show(false)

    val indexToString = new IndexToString()
      .setInputCol("colorIndex")
      .setOutputCol("originalColor")

    val df3 = indexToString.transform(df2)
    df3.show(false)

    spark.stop
  }
  
}