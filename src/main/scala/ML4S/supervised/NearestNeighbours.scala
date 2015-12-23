package ML4S.supervised

import breeze.linalg.{argtopk, *, DenseMatrix, DenseVector}


class NearestNeighbours(k: Int,
                        dataX: DenseMatrix[Double],
                        dataY: Seq[String],
                        distanceFn: (DenseVector[Double],DenseVector[Double]) => Double) {

  def predict(x: DenseVector[Double]):String={

    //compute similarity for each example.
    val distances = dataX(*,::)
      .map(r => distanceFn(r,x))

    //Get top k most similar classes
    val topKClasses=distances
      .toArray
      .zipWithIndex
      .sortBy(_._1)
      .take(k)
      .map{ case(dist,idx) =>  dataY(idx)}

    println("top k classes: "+topKClasses.toList)
    //Most frequent class in top K
    topKClasses
    .groupBy(identity)
    .mapValues(_.size)
    .maxBy(_._2)._1
  }


}
