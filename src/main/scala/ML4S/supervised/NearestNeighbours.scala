package ML4S.supervised

import breeze.linalg.{*, DenseMatrix, DenseVector}

/**
 * An implementation of the k-nearest neighbours classification algorithm.
 * @param k The number of neighbours to use for prediction.
 * @param dataX Matrix of input examples.
 * @param dataY Corresponding output classes.
 * @param distanceFn Function used to compute 'near-ness'.
 */
class NearestNeighbours(k: Int,
                        dataX: DenseMatrix[Double],
                        dataY: Seq[String],
                        distanceFn: (DenseVector[Double], DenseVector[Double]) => Double) {

  /**
   * Predict the output class corresponding to a given input example
   * @param x input example
   * @return predicted class
   */
  def predict(x: DenseVector[Double]): String = {

    //compute similarity for each example.
    val distances = dataX(*, ::)
      .map(r => distanceFn(r, x))

    //Get top k most similar classes
    val topKClasses = distances
      .toArray
      .zipWithIndex
      .sortBy(_._1)
      .take(k)
      .map { case (dist, idx) => dataY(idx)}

    //Most frequent class in top K
    topKClasses
      .groupBy(identity)
      .mapValues(_.size)
      .maxBy(_._2)._1
  }


}
