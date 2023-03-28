case class iris_in_data(
                         sepal_length: Double,
                         sepal_width: Double,
                         petal_length: Double,
                         petal_width: Double
                       )
object iris_in_data {
  def apply(a: Array[String]): iris_in_data =
    iris_in_data(
      a(0).toDouble,
      a(1).toDouble,
      a(2).toDouble,
      a(3).toDouble
    )
}
