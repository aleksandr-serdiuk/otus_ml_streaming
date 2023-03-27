import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.producer.ProducerRecord
import org.apache.kafka.common.serialization.{IntegerSerializer, StringSerializer}

import java.util.Properties
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.types.{ArrayType, DoubleType, IntegerType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.kafka.clients.consumer.ConsumerConfig
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.kafka.common.serialization.{StringDeserializer, StringSerializer}
import org.apache.spark.streaming.kafka010._
import org.apache.spark.sql.functions._


object iris_streaming {
  def main(args: Array[String]): Unit = {

/*
  kafka-topics.sh --bootstrap-server localhost:29092 --topic irisin --create
  kafka-topics.sh --bootstrap-server localhost:29092 --topic irisout --create
  kafka-topics.sh --bootstrap-server localhost:29092 --list

  kafka-console-producer.sh --bootstrap-server localhost:29092 --topic irisin

  kafka-console-consumer.sh --bootstrap-server localhost:29092 --topic irisout
  kafka-console-consumer.sh --bootstrap-server localhost:29092 --topic irisout --formatter kafka.tools.DefaultMessageFormatter --property print.timestamp=true --property print.key=true --property print.value=true --property print.partition=true --from-beginning
 */

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

    spark.close()

    // Создаём Streaming Context и получаем Spark Context
    val sparkConf = new SparkConf().setAppName("iris_streaming").setMaster("local")
    val streamingContext = new StreamingContext(sparkConf, Seconds(1))

    // Загружаем модель
    val iris_rf_model = PipelineModel.load("src/main/resources/iris_rf.model")

    // Создаём свойства Producer'а для вывода в выходную тему Kafka (тема с расчётом)
    val props: Properties = new Properties()
    props.put("bootstrap.servers", "localhost:29092")

    // Параметры подключения к Kafka для чтения
    val kafkaParams = Map[String, Object](
      ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG -> "localhost:29092",
      ConsumerConfig.GROUP_ID_CONFIG -> "iris",
      ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG -> classOf[StringDeserializer],
      ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG -> classOf[StringDeserializer]
    )

    // Подписываемся на входную тему Kafka (тема с данными)
    val inputTopic = "irisin"
    val inputTopicSet = Set(inputTopic)
    val messages = KafkaUtils.createDirectStream[String, String](
      streamingContext,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](inputTopicSet, kafkaParams)
    )

    // Разбиваем входную строку на элементы
    val lines = messages
      .map(_.value)
      .map(_.replace("\"", "").split(","))

    // Обрабатываем каждый входной набор
    lines.foreachRDD { rdd =>
      if (!rdd.isEmpty) {
        val spark = SparkSession.builder.config(rdd.sparkContext.getConf).getOrCreate()
        import spark.implicits._

        // Преобразовываем RDD в DataFrame
        val data = rdd
          .toDF("input")
          .withColumn("sepal_length", $"input"(0).cast(DoubleType))
          .withColumn("sepal_width", $"input"(1).cast(DoubleType))
          .withColumn("petal_length", $"input"(2).cast(DoubleType))
          .withColumn("petal_width", $"input"(3).cast(DoubleType))
          .drop("input")

        val featureColumns: Array[String] = Array(
          "sepal_length",
          "sepal_width",
          "petal_length",
          "petal_width"
        )

        val assembler = new VectorAssembler()
          .setInputCols(featureColumns)
          .setOutputCol("features")

        val assembled = assembler.transform(data).select("features")
        //assembled.show(5, truncate = false)

        val predictions = iris_rf_model.transform(assembled)

        def getFeature = udf((featureV: org.apache.spark.ml.linalg.Vector, clsInx: Int) => featureV.apply(clsInx))

        val resultDF = predictions
          .join(dictIris, $"iris_id" === $"predictedLabel")
          .withColumn("sepal_length", getFeature($"features", lit(0)))
          .withColumn("sepal_width", getFeature($"features", lit(1)))
          .withColumn("petal_length", getFeature($"features", lit(2)))
          .withColumn("petal_width", getFeature($"features", lit(3)))
          .withColumn("result", concat_ws(",", col("sepal_length"), col("sepal_width"), col("petal_length"), col("petal_width"), col("predictedLabel"), col("iris_value")))
          .select("result")

        resultDF.show(false)

        val props = new Properties()
        props.put("bootstrap.servers", "localhost:29092")
        val producer = new KafkaProducer(props, new StringSerializer, new StringSerializer)
        val output_topic = "irisout"

        resultDF
          .rdd
          .foreachPartition { partition =>
            val producer = new KafkaProducer(props, new StringSerializer, new StringSerializer)
            partition.foreach { row =>
              producer.send(new ProducerRecord(output_topic, s"${row.getString(0)}"))
            }
            producer.close()
          }

      }
    }


    streamingContext.start()
    streamingContext.awaitTermination()

  }
}
