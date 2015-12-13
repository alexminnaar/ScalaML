package ML4S.examples

import breeze.linalg.DenseVector
import breeze.plot._

import scala.io.Source


object ServiceRequestDataViz extends App{

  def toDouble(s: String): Option[Double] = {
    try {
      Some(s.toDouble)
    } catch {
      case e: Exception => None
    }
  }


  val dataset = Source.fromFile("datasets/311_Service_Requests_for_2009.csv")
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


  val x = dataset.map(x => x(0)).toIndexedSeq
  val y = dataset.map(x => x(1)).toIndexedSeq

  f.subplot(0) +=  scatter(x, y, {(_:Int) => 100})
  f.subplot(0).xlabel = "X-coordinate"
  f.subplot(0).ylabel = "Y-coordinate"
  f.subplot(0).title = "311 Service Noise Complaints"



}
