package ML4S.examples

import ML4S.akka.NNMaster
import ML4S.akka.NNMaster.{Prediction, QueryInput}
import akka.actor.{ActorSystem, Props, Actor}
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source


class AkkaNNExample extends Actor{

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

  val inputs = data.map(r => DenseVector(r._1.toArray))


  val euclideanDist = (v1: DenseVector[Double], v2: DenseVector[Double])
  =>
    v1
      .toArray
      .zip(v2.toArray)
      .map(x => math.pow((x._1 - x._2), 2))
      .sum


  val exampleNNMaster = context.actorOf(Props(new NNMaster(
    inputs,
    outputs,
    4,
    euclideanDist,
    4
  )))

  exampleNNMaster ! QueryInput(
    DenseVector(1,0,1,0.08380,1,0.17387,1,-0.13308,0.98172,0.64520,1,0.47904,1,0.59113,1,0.70758,1,0.82777,1,0.95099,1,1,0.98042,1,0.91624,1,0.83899,1,0.74822,1,0.64358,1,0.52479,1)
  )

  def receive = {
    case Prediction(p) => println(s"predicted output is class: ${p}")
  }


}



object Driver {

  def main(args: Array[String]) {
    val system = ActorSystem("Main")
    val ac = system.actorOf(Props[AkkaNNExample])
  }

}