package ML4S.examples

import ML4S.supervised.NearestNeighbours
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source


object NNExample extends App {


  def line2Data(line: String): (List[Double], String) = {

    val elements = line.split(",")

    val y = elements.last

    val x = elements
      .dropRight(1)
      .map(_.toDouble)
      .toList

    (x, y)
  }


  val data = Source.fromFile("datasets/ionosphere.data")
    .getLines()
    .map(x => line2Data(x))
    .toList

  val outputs = data.map(_._2).toSeq

  val inputs = DenseMatrix(data.map(_._1).toArray: _*)

  val euclideanDist = (v1: DenseVector[Double], v2: DenseVector[Double])
  =>
    v1
      .toArray
      .zip(v2.toArray)
      .map(x => math.pow((x._1 - x._2), 2))
      .sum


  val trainInputs = inputs(0 to 299, ::)
  val trainOutputs = outputs.take(300)


  val myNN = new NearestNeighbours(
    k = 4,
    dataX = trainInputs,
    dataY = trainOutputs,
    euclideanDist
  )

  var correct = 0

  (300 to 350).foreach { exampleId =>

    val pred = myNN.predict(inputs(exampleId, ::).t)
    val target = outputs(exampleId)

    if (pred == target) correct += 1

  }

  println(correct.toDouble / (300 to 350).length)

}
