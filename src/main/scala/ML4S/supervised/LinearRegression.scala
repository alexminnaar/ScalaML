package ML4S.supervised

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import breeze.stats.mean


/**
 * Class for a linear regression supervised learning model.
 * @param inputs A matrix whose rows are the input vectors corresponding to each training example.
 * @param outputs A matrix whose rows are the outputs corresponding to each training example
 * @param basisFn An optional basis function to be applied to the inputs (for generalized linear models).
 */
class LinearRegression(inputs: DenseMatrix[Double],
                       outputs: DenseMatrix[Double],
                       basisFn: Option[DenseVector[Double] => DenseVector[Double]] = None) {

  //If a basis function has been provided, apply it to each input example
  val x = basisFn match {
    case Some(bf) => inputs(*, ::).map(dv => bf(dv))
    case None => inputs
  }

  /**
   * Given an input example vector and a weight vector, predict the output
   * @param weights Learning LR weight vector.
   * @param input Input example vector.
   * @return Prediction.
   */
  def predict(weights: DenseMatrix[Double],
              input: DenseMatrix[Double]): DenseMatrix[Double] = {
    input * weights
  }

  /**
   * Train a weight vector for a LR model.
   * @param inputs The input training examples, by default they are the ones provided in the constructor
   * @param outputs The output training examples, by default they are the ones provided in the constructor.
   * @param regularizationParam The regularization penalty weight, by default it is zero (no regularization).
   * @return A weight vector.
   */
  def train(inputs: DenseMatrix[Double] = x,
            outputs: DenseMatrix[Double] = outputs,
            regularizationParam: Double = 0.0): DenseMatrix[Double] = {

    val l = inputs.cols

    val identMat = DenseMatrix.eye[Double](l)
    val regPenalty = regularizationParam * l

    //The normal equation for LR (with regularization)
    inv(inputs.t * inputs + regPenalty * identMat) * (inputs.t * outputs)
  }

  /**
   * Compute the MSE for a LR model on test data.
   * @param weights Weight vector for a learning LR model.
   * @param inputs Inputs for test data.
   * @param targets Outputs for test data.
   * @return MSE.
   */
  def evaluate(weights: DenseMatrix[Double],
               inputs: DenseMatrix[Double],
               targets: DenseMatrix[Double],
               evaluator: (DenseMatrix[Double], DenseMatrix[Double]) => Double): Double = {

    //compute predictions
    val preds = predict(weights, inputs)

    //compare predictions to targets using MSE
    evaluator(preds, targets)
  }

  /**
   * Perform k-fold cross-validation using the entire dataset provided in constructor.
   * @param folds The number of cross-validation folds to use.
   * @param regularizationParam The regularization parameter to use.
   * @return The average cross-validation error over all folds.
   */
  def crossValidation(folds: Int,
                      regularizationParam: Double,
                      evaluator: (DenseMatrix[Double], DenseMatrix[Double]) => Double): Double = {

    val foldSize = x.rows / folds.toDouble

    //segment dataset
    val partitions = (0 to x.rows - 1).grouped(math.ceil(foldSize).toInt)

    val ptSet = (0 to x.rows - 1).toSet

    //compute test error for each fold
    val xValError = partitions.foldRight(Vector.empty[Double]) { (c, acc) =>

      //training data points are all data points not in validation set.
      val trainIdx = ptSet.diff(c.toSet)
      val testIdx = c

      //training data
      val trainX = x(trainIdx.toIndexedSeq, ::).toDenseMatrix
      val trainY = outputs(trainIdx.toIndexedSeq, ::).toDenseMatrix

      //test data
      val testX = x(testIdx.toIndexedSeq, ::).toDenseMatrix
      val testY = outputs(testIdx.toIndexedSeq, ::).toDenseMatrix

      //train a weight vector with the above training data
      val weights = train(trainX, trainY, regularizationParam)

      //compute the error on the held-out test data
      val error = evaluate(weights, testX, testY, evaluator)

      //append error to the accumulator so it can be average later
      acc :+ error
    }

    mean(xValError)
  }


}
