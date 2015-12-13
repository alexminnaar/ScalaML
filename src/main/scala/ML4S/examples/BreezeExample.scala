package ML4S.examples

import breeze.linalg.DenseMatrix


object BreezeExample extends App{

  //create a denseMatrix
  val dm1 = DenseMatrix((1.0,2.0),(3.0,4.0))

  //matrix transpose
  val dm1Transpose = dm1.t

  println(s"${dm1} transposed is ${dm1Transpose}")

  //create a second denseMatrix of the same size
  val dm2 = DenseMatrix((5.0,6.0),(7.0,8.0))

  //matrix product
  val matrixProduct = dm1*dm2

  println(s"The product of ${dm1} and ${dm2} is ${matrixProduct}")

  //matrix elementwise sum
  val matrixElSum = dm1:+dm2

  println(s"The elementwise sum of ${dm1} and ${dm2} is ${matrixElSum}")

}
