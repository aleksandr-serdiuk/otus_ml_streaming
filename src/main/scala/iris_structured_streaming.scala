import com.typesafe.config._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.concat_ws
import ru.otus.sparkml.Data
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.functions._


object iris_structured_streaming {
  def main(args: Array[String]): Unit = {
/*
    kafka-topics.sh --bootstrap-server localhost:29092 --topic irisin_structured --create
    kafka-topics.sh --bootstrap-server localhost:29092 --topic irisout_structured --create
    kafka-topics.sh --bootstrap-server localhost:29092 --list

    kafka-console-producer.sh --bootstrap-server localhost:29092 --topic irisin_structured
    kafka-console-consumer.sh --bootstrap-server localhost:29092 --topic irisout_structured

*/

    val inputTopic             = "irisin_structured"
    val outputTopic            = "irisout_structured"

    // Создаём SparkSession
    val spark = SparkSession.builder
      .appName("MLStructuredStreaming")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    // Читаем входной поток
    val input = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:29092")
      .option("subscribe", inputTopic)
      .option("failOnDataLoss", false)
      .load()
      .selectExpr("CAST(value AS STRING)")
      .as[String]
      .map(_.replace("\"", "").split(","))
      .map(iris_in_data(_))

    // Загружаем модель
    val iris_rf_model = PipelineModel.load("src/main/resources/iris_rf.model")

    val featureColumns: Array[String] = Array(
      "sepal_length",
      "sepal_width",
      "petal_length",
      "petal_width"
    )

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val assembled = assembler.transform(input).select("features")

    val predictions = iris_rf_model.transform(assembled)

    def getFeature = udf((featureV: org.apache.spark.ml.linalg.Vector, clsInx: Int) => featureV.apply(clsInx))

    val dictIris = spark.createDataFrame(Seq(
      (0, "Iris-setosa"),
      (1, "Iris-versicolor"),
      (2, "Iris-virginica")
    )).toDF("iris_id", "iris_value")

    val resultDF = predictions
      .join(dictIris, $"iris_id" === $"predictedLabel")
      .withColumn("sepal_length", getFeature($"features", lit(0)))
      .withColumn("sepal_width", getFeature($"features", lit(1)))
      .withColumn("petal_length", getFeature($"features", lit(2)))
      .withColumn("petal_width", getFeature($"features", lit(3)))
      .withColumn("value", concat_ws(",", col("sepal_length"), col("sepal_width"), col("petal_length"), col("petal_width"), col("predictedLabel"), col("iris_value")))
      .select("value")

    // Выводим результат
    val query = resultDF
      .writeStream
      .option("checkpointLocation", "src/main/resources/cpl/")
      .outputMode("append")
      .format("kafka")
      .option("failOnDataLoss", false)
      .option("kafka.bootstrap.servers", "localhost:29092")
      .option("topic", outputTopic)
      .start()

    query.awaitTermination()

  }
}

