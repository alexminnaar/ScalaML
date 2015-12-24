package ML4S.supervised

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import breeze.numerics.pow
import breeze.stats.mean


class LinearRegression(inputs: DenseMatrix[Double],
                       outputs: DenseMatrix[Double],
                       basisFn: Option[DenseVector[Double] => DenseVector[Double]] = None) {

  val x = basisFn match {
    case Some(bf) => inputs(*, ::).map(dv => bf(dv))
    case None => inputs
  }

  def predict(weights: DenseMatrix[Double],
              input: DenseMatrix[Double]): DenseMatrix[Double] = {

    input * weights
  }

  def train(inputs: DenseMatrix[Double] = x,
            outputs: DenseMatrix[Double] = outputs,
            regularizationParam: Double = 0.0): DenseMatrix[Double] = {

    val l = inputs.cols

    val identMat = DenseMatrix.eye[Double](l)
    val regPenalty = regularizationParam * l

    inv(inputs.t * inputs + regPenalty * identMat) * (inputs.t * outputs)
  }

  def evaluate(weights: DenseMatrix[Double],
               inputs: DenseMatrix[Double],
               targets: DenseMatrix[Double]): Double = {

    val preds = predict(weights, inputs)

    mean((preds - targets).map(x => pow(x, 2)))
  }


  def crossValidation(folds: Int,
                      regularizationParam: Double): Double = {

    val partitions = (0 to x.rows - 1).grouped(folds)

    val ptSet = (0 to x.rows - 1).toSet

    val xValError = partitions.foldRight(Vector.empty[Double]) { (c, acc) =>

      val trainIdx = ptSet.diff(c.toSet)
      val testIdx = c

      val trainX = x(trainIdx.toIndexedSeq, ::).toDenseMatrix
      val trainY = outputs(trainIdx.toIndexedSeq, ::).toDenseMatrix

      val testX = x(testIdx.toIndexedSeq, ::).toDenseMatrix
      val testY = outputs(testIdx.toIndexedSeq, ::).toDenseMatrix

      val weights = train(trainX, trainY, regularizationParam)

      val error = evaluate(weights, testX, testY)

      acc :+ error
    }

    mean(xValError)
  }


}
