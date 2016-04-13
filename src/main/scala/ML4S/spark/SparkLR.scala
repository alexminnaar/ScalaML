package ML4S.spark

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}

import scala.io.Source

/**
  * Spark linear regression.  This example uses code from the Spark documentation example found at
  * http://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-least-squares-lasso-and-ridge-regression
  */
object SparkLR extends App {

  def line2Data(line: String): Array[Double] = {

    line
      .split("\\s+")
      .filter(_.length > 0)
      .map(_.toDouble)
  }

  //import data
  val data = Source.fromFile("datasets/boston_housing.data")
    .getLines()
    .map { l =>
      val formattedRow = line2Data(l)

      val input = formattedRow.dropRight(1)
      val output = formattedRow.last

      LabeledPoint(output, Vectors.dense(input))
    }.toArray


  val conf = new SparkConf()
    .setAppName("Spark Linear Regression")
    .setMaster("local[3]")

  val sc = new SparkContext(conf)


  //parallelize the data
  val dataRDD = sc.parallelize(data)


  //model parameters
  val numIterations = 100
  val stepSize = 0.000001

  val model = LinearRegressionWithSGD.train(dataRDD, numIterations, stepSize)


  //make predictions from the model
  val predsAndActual = dataRDD.map { example =>

    val pred = model.predict(example.features)

    (example.label, pred)
  }


  val mse = predsAndActual.map { case (actual, pred) =>
    math.pow((actual - pred), 2)
  }.mean()


  println(s"training error: ${mse}")

}
