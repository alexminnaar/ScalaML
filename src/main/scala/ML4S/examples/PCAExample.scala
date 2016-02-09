package ML4S.examples

import ML4S.unsupervised.PrincipleComponentsAnalysis
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source


object PCAExample extends App{

  def line2Data(line: String): Array[Double] = {

    line
      .split("\\s+")
      .filter(_.length > 0)
      .map(_.toDouble)

  }

  //import data
  val data = Source.fromFile("datasets/boston_housing.data")
    .getLines()
    .map(x => line2Data(x))
    .toArray

  //convert to breeze matrix
  val dm = DenseMatrix(data: _*)

  //the inputs are all but the last column.  Outputs are last column
  val X = dm(::, 0 to 12)

  val pca = new PrincipleComponentsAnalysis(dm)

  println(pca.componentVariance)

  println(pca.transformedData)



}


