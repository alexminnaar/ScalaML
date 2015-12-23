package ML4S.examples

import ML4S.supervised.LinearRegression
import breeze.linalg.DenseMatrix

import scala.io.Source


object LrExample extends App {


  def line2Data(line: String): Array[Double] = {

    line
      .split("\\s+")
      .filter(_.length > 0)
      .map(_.toDouble)

  }


  val data = Source.fromFile("datasets/boston_housing.data")
    .getLines()
    .map(x => line2Data(x))
    .toArray

  val dm = DenseMatrix(data: _*)

  val X = dm(::, 0 to 12)
  val y = dm(::, -1).toDenseMatrix.t

  //println(outputs)

  val myLr = new LinearRegression(inputs = X,
  outputs = y)

  val weights = myLr.train(
  regularizationParam = 0.00001)

  val example = X(0,::).t.toDenseMatrix

  val pred = myLr.predict(weights,example)



  println(weights)

println(pred)

}
