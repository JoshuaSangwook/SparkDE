package mlBasic

import org.apache.spark.ml.classification.{ LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object ClassificationBasic {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder()
      .appName("ClassficationSample")
      .master("local[*]")
      .getOrCreate()

    val training = spark.createDataFrame(Seq(
      (161.0, 69.87, 29, 1.0),
      (176.78, 74.35, 34, 1.0),
      (159.23, 58.32, 29, 0.0))).toDF("height", "weight", "age", "gender")

    val test = spark.createDataFrame(Seq(
      (169.4, 75.3, 42),
      (185.1, 85.0, 37),
      (161.6, 61.2, 28))).toDF("height", "weight", "age")

    training.show(false)
    test.show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("height", "weight", "age"))
      .setOutputCol("features")

    // training 데이터에 features 컬럼 추가
    val assembled_training = assembler.transform(training)

    assembled_training.show(false)

    // 모델 생성 알고리즘 (로지스틱 회귀 평가자)
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setLabelCol("gender")

    // 모델 생성
    val model = lr.fit(assembled_training)

    // 예측값 생성
    model.transform(assembled_training).show()

    // test data 로 진
    val assembled_test = assembler.transform(test)
    assembled_test.show()
    model.transform(assembled_test).show()

  }

}