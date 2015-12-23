package ML4S.supervised

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}


class LinearRegression(inputs: DenseMatrix[Double],
                       outputs: DenseMatrix[Double],
                       basisFn: Option[DenseVector[Double] => DenseVector[Double]] = None) {

  val x = basisFn match {
    case Some(bf) => inputs(*, ::).map(dv => bf(dv))
    case None => inputs
  }

  def predict(weights:DenseMatrix[Double],
              input: DenseMatrix[Double]): DenseMatrix[Double] = {

    input * weights
  }

  def train(inputs: DenseMatrix[Double] = x,
            outputs: DenseMatrix[Double] = outputs,
            regularizationParam: Double = 0.0): DenseMatrix[Double] = {

    val l = inputs.cols

    val identMat = DenseMatrix.eye[Double](l)
    val regPenalty = regularizationParam * l

    inv(x.t * x + regPenalty * identMat) * (x.t * outputs)
  }


}
