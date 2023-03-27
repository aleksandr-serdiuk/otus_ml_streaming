import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer, VectorSlicer}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._

object iris_model_test extends App {

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
    .option("header", true)
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

  import spark.implicits._
  predictions
    .withColumn("newCol",  $"features")
    .show(false)

  def getFeature = udf((featureV: org.apache.spark.ml.linalg.Vector, clsInx: Int) => featureV.apply(clsInx))

  val resultDF = predictions
    .join(dictIris, $"iris_id" === $"predictedLabel")
    .withColumn("sepal_length", getFeature($"features", lit(0)))
    .withColumn("sepal_width", getFeature($"features", lit(1)))
    .withColumn("petal_length", getFeature($"features", lit(2)))
    .withColumn("petal_width", getFeature($"features", lit(3)))
    .withColumn("result_csv", concat_ws(",", col("sepal_length"), col("sepal_width"), col("petal_length"), col("petal_width"), col("predictedLabel"), col("iris_value")))
    .select("result_csv")

  resultDF.show(false)
  resultDF.printSchema()

  /*
  val slicer = new VectorSlicer()
    .setInputCol("features")
    .setIndices(Array(0))
    .setOutputCol("sepal_length")

  val posi_output = slicer.transform(resultDF)
  posi_output.show(false)
*/

  resultDF
    .foreach(row => {
      //println(s"Prediction: ${row.getDouble(0)},${row.getDouble(1)},${row.getDouble(2)},${row.getDouble(3)},${row.getString(4)},${row.getString(5)}")
      println(s"${row.getString(0)}")
    })

  spark.close()


}
