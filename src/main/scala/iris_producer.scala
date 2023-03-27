import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.producer.ProducerRecord
import org.apache.kafka.common.serialization.{IntegerSerializer, StringSerializer}

import java.util.Properties
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.types.{ArrayType, DoubleType, IntegerType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object iris_producer extends App {

  val spark = SparkSession
    .builder()
    .appName("ML_test")
    .config("spark.master", "local")
    .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
    .getOrCreate()

  val dictIris = spark.createDataFrame(Seq(
    (0, "Iris-setosa"),
    (1, "Iris-versicolor"),
    (2, "Iris-virginica")
  )).toDF("iris_id", "iris_value")

  val iris_rf_model = PipelineModel.load("src/main/resources/iris_rf.model")

  val schema_scv = new StructType()
    .add("sepal_length", DoubleType, true)
    .add("sepal_width", DoubleType, true)
    .add("petal_length", DoubleType, true)
    .add("petal_width", DoubleType, true)
    .add("species", StringType, true)

  val df_csv = spark.read.format("csv")
    .option("header",true)
    .schema(schema_scv)
    .csv("src/main/resources/IRIS_2.csv")

  val featureColumns: Array[String] = Array(
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
  )

  val assembler = new VectorAssembler()
    .setInputCols(featureColumns)
    .setOutputCol("features")

  val assembled = assembler.transform(df_csv).select("features")
  assembled.show(5, truncate = false)

  val predictions = iris_rf_model.transform(assembled)

  predictions.show(false)

  import spark.implicits._
  val resultDF = predictions
    .join(dictIris, $"iris_id" === $"predictedLabel")
    .select("predictedLabel", "iris_value")

  resultDF.show(false)

  val props = new Properties()
  props.put("bootstrap.servers", "localhost:29092")
  val producer = new KafkaProducer(props, new StringSerializer, new StringSerializer)
  val topic_name = "iris"

  resultDF
    .foreach(row => {
      producer.send(new ProducerRecord(topic_name, s"Prediction: ${row.getString(0)} - ${row.getString(1)}"))
      println(s"Prediction: ${row.getString(0)} - ${row.getString(1)}")
    })

  producer.close()
  spark.close()


}
