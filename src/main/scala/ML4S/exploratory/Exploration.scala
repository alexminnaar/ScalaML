package ML4S.exploratory

import breeze.plot._
import org.saddle.{Frame, Mat, Vec}

import scala.io.Source

object Exploration extends App {

  def line2Data(line: String): List[Double] = {

    line
      .split("\\s+")
      .filter(_.length > 0)
      .map(_.toDouble)
      .toList
  }

  def row2Vec(row: List[Double]): Vec[Double] = Vec(row: _*)

  val data = Source.fromFile("datasets/boston_housing.data")
    .getLines()
    .map(x => line2Data(x))

  val vecs = data.map(r => row2Vec(r))

  val matr = Mat(vecs.toArray)

  val fr = Frame(matr.transpose)

  val output = fr.colAt(13)
  println("mean: "+output.mean)
  println("median: "+output.median)
  println("variance: "+output.variance)
  println("maximum: "+output.max)
  println("minimum: "+output.min)

  val f = Figure()
  f.width = 800
  f.height = 800

  val columns = Vector("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")

  //val xs = for (x <- fr.colAt(0).toSeq) yield x._2
  val ys = for (y <- fr.colAt(13).toSeq) yield y._2

  println(ys)
 // f.subplot(0)

  //f.subplot(0) += hist(ys,10)



  val subplots = for (i <- 0 to 3; j <- 0 to 3)
  yield { f.subplot(4, 4, i + 4 * j)  }


  for (i <- 0 to 12; j <- 0 to 3) {
    val xs = for (x <- fr.colAt(i).toSeq) yield { x._2 }
    val p = subplots(i)
    p += plot(xs, ys, '+')
    p.xlabel = columns(i)
    p.ylabel = "PRICE"
  }



}
