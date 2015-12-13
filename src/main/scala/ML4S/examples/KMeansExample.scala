package ML4S.examples

import java.awt.{Color, Paint}

import ML4S.unsupervised.Kmeans
import breeze.linalg.{DenseVector, sum}
import breeze.plot._

import scala.io.Source

object KMeansExample extends App {

  def toDouble(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case e: Exception => None
    }
  }


  val SRDataset = Source.fromFile("datasets/311_Service_Requests_for_2009.csv")
    .getLines()
    .map(line => line.split(","))
    .filter(_(5) == "Noise")
    .filter { splitLine =>

    splitLine.length match {
      case 53 => (toDouble(splitLine(24)) != None) && (toDouble(splitLine(25)) != None)
      case 54 => (toDouble(splitLine(25)) != None) && (toDouble(splitLine(26)) != None)
      case _ => false
    }
  }
    .map { splitLine =>

    if (splitLine.length == 53) DenseVector(splitLine(24).toDouble, splitLine(25).toDouble)
    else DenseVector(splitLine(25).toDouble, splitLine(26).toDouble)

  }
    .toSeq


  val f = Figure()

  val euclideanDistance =
    (dp1: DenseVector[Double], dp2: DenseVector[Double]) =>
      sum((dp1 - dp2).map(el => el * el))


  val clusters = Kmeans.cluster(dataset = SRDataset,
    numClusters = 6,
    distanceFunc = euclideanDistance)


  val id2Color: Int => Paint = id => id match {
    case 0 => Color.YELLOW
    case 1 => Color.RED
    case 2 => Color.GREEN
    case 3 => Color.BLUE
    case 4 => Color.GRAY
    case _ => Color.BLACK
  }

  f.subplot(0).xlabel = "X-coordinate"
  f.subplot(0).ylabel = "Y-coordinate"
  f.subplot(0).title = "311 Service Noise Complaints"

  clusters.zipWithIndex.foreach { case (cl, clIdx) =>

    val clusterX = clusters(clIdx).assignedDataPoints.map(_(0))
    val clusterY = clusters(clIdx).assignedDataPoints.map(_(1))
    f.subplot(0) += scatter(clusterX, clusterY, { (_: Int) => 1000}, { (_: Int) => id2Color(clIdx)})
  }


}
