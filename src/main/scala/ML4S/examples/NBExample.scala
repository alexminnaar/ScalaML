package ML4S.examples

import ML4S.supervised.NaiveBayes

import scala.io.Source
import breeze.linalg._

object NBExample extends App {

  def row2Data(row: Seq[String]): (Seq[Double], String) = {

    val x = row
      .dropRight(1)
      .map(_.toDouble)

    val y = row.last

    (x, y)
  }

  val dataset = Source
    .fromFile("datasets/ionosphere.data")
    .getLines()
    .map(r => row2Data(r.split(",")))
    .toList


  val inputs = DenseMatrix(dataset.map(_._1): _*)

  val outputs = dataset.map(_._2)


  val myNB = new NaiveBayes(
    dataX = inputs,
    dataY = outputs
  )

  var correctCounter = 0

  for(i <- 0 to outputs.length - 1){

    val prediction = myNB.predict(inputs(i,::).t)
    val actual = outputs(i)

    println(prediction,actual)

    if(prediction == actual) correctCounter+=1

  }

  println(correctCounter)
  println(correctCounter.toDouble/outputs.length)

}