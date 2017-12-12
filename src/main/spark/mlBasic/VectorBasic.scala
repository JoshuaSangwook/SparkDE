package mlBasic

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

object VectorBasic {
    def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .appName("VectorSample")
      .master("local[*]")
      .getOrCreate()

    val v1 = Vectors.dense(0.1, 0.0, 0.2, 0.3);
    val v2 = Vectors.dense(Array(0.1, 0.0, 0.2, 0.3))
    val v3 = Vectors.sparse(4, Seq((0, 0.1), (2, 0.2), (3, 0.3)))
    val v4 = Vectors.sparse(4, Array(0, 2, 3), Array(0.1, 0.2, 0.3))

    // 8.1.1절
    println(v1.toArray.mkString(", "))
    println(v2.toArray.mkString(", "))
    println(v3.toArray.mkString(", "))
    println(v4.toArray.mkString(", "))
    }
}