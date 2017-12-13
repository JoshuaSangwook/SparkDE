package mlBasic

//import org.apache.log4j.{ Level, Logger }
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ RandomForestClassificationModel, RandomForestClassifier }
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

object ProblemFourRandomForestClassifier {

  //Logger.getLogger("org").setLevel(Level.OFF)
  //Logger.getLogger("akka").setLevel(Level.OFF)



  def main(args: Array[String]) {
    
      var inputDir = "./data/moviereviews.tsv"
  var stopWordFile = "./data/stopwords.txt"

  val vocabSize: Int = 10000

    val spark = SparkSession
      .builder()
      .appName("ClassficationSample")
      .master("local[*]")
      .config("spark.driver.host", "127.0.0.1")
      .getOrCreate()

    val df = spark.read.option("header", "true").
      option("sep", "\t").
      csv(inputDir)
      
    //df.show()  
    val cleanData = udf((sentiment: String) => sentiment.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", ""))
    val cleanDF = df.withColumn("cleanedReview", cleanData(df.col("review")))
    //cleanDF.show()  
    
    val tokens = new RegexTokenizer().
      setGaps(false).
      setPattern("\\w+").
      setMinTokenLength(4).
      setInputCol("cleanedReview").
      setOutputCol("words").
      transform(cleanDF)
      
     tokens.show() 
     
     val stopwords = spark.sparkContext.textFile(stopWordFile).collect
     
     val filteredTokens = new StopWordsRemover().
      setStopWords(stopwords).
      setCaseSensitive(false).
      setInputCol("words").
      setOutputCol("filtered").
      transform(tokens)
      
        val cvModel = new CountVectorizer().
      setInputCol("filtered").
      setOutputCol("features").
      setVocabSize(vocabSize).
      fit(filteredTokens)
      
      val countVectors = cvModel.
      transform(filteredTokens).
      select("id", "sentiment", "features")
      
      countVectors.show()
      
       val labelIndexer = new StringIndexer().
      setInputCol("sentiment").
      setOutputCol("indexedLabel").
      fit(countVectors)

  val featureIndexer = new VectorIndexer().
      setInputCol("features").
      setOutputCol("indexedFeatures").
      setMaxCategories(4).
      fit(countVectors)
      
      val Array(trainingData, testData) = countVectors.randomSplit(Array(0.7, 0.3))

  val rf = new RandomForestClassifier().
      setLabelCol("indexedLabel").
      setFeaturesCol("indexedFeatures").
      setNumTrees(200)

  val labelConverter = new IndexToString().
      setInputCol("prediction").
      setOutputCol("predictedLabel").
      setLabels(labelIndexer.labels)

      val pipeline = new Pipeline().
      setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  /**
    * Train model. This also runs the indexers.
    */
  val model = pipeline.fit(trainingData)

  /**
    * Make predictions
    */
  val predictions = model.transform(testData)

  /**
    * Display Result
    */
  predictions.select("predictedLabel", "sentiment", "features").show(30)
      
   val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("accuracy")
  
  val accuracy = evaluator.evaluate(predictions)


  /**
    * Print the error
    */
  println("Test Error = " + (1.0 - accuracy))
  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println("Learned classification forest model:\n" + rfModel.toDebugString)
  }
}