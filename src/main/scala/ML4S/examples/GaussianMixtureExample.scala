package ML4S.examples

import ML4S.unsupervised.GaussianMixture
import breeze.linalg.DenseMatrix

import scala.io.Source


object GaussianMixtureExample extends App {


  def toDouble(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case e: Exception => None
    }
  }

  val srDataset = Source.fromFile("datasets/311_Service_Requests_for_2009.csv")
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

    if (splitLine.length == 53) Array(splitLine(24).toDouble, splitLine(25).toDouble)
    else Array(splitLine(25).toDouble, splitLine(26).toDouble)

  }
    .toSeq


  // srDataset.foreach(println)

  val dm = DenseMatrix(srDataset: _*)


  val gmm = new GaussianMixture(
    dataPoints = dm,
    numClusters = 5
  )


  gmm.cluster()

}
