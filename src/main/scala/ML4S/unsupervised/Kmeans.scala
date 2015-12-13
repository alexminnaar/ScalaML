package ML4S.unsupervised

import breeze.linalg.DenseVector
import scala.util.Random

case class Cluster(mean: DenseVector[Double], assignedDataPoints: Seq[DenseVector[Double]])

object Kmeans {

  def initializeClusters(dataSet: Seq[DenseVector[Double]],
                         numClusters: Int): Seq[Cluster] = {

    val dataDim = dataSet.head.length
    val randomizedData = Random.shuffle(dataSet)
    val groupSize = math.ceil(dataSet.size / numClusters.toDouble).toInt

    randomizedData
      .grouped(groupSize)
      .map(group => Cluster(DenseVector.zeros[Double](dataDim), group))
      .toSeq
  }

  def computeMean(data: Seq[DenseVector[Double]]): DenseVector[Double] = {

    val dataDim = data.head.length

    val meanArray = data.foldLeft(Array.fill[Double](dataDim)(0.0)) { (acc, dataPoint) =>
      (acc, dataPoint.toArray).zipped.map(_ + _)
    }.map(_ / data.size)

    DenseVector(meanArray)
  }


  def assignDataPoints(clusterMeans: Seq[DenseVector[Double]],
                       dataPoints: Seq[DenseVector[Double]],
                       distance: (DenseVector[Double], DenseVector[Double]) => Double): Seq[Cluster] = {

    val dataDim = dataPoints.head.length

    var initialClusters = Map.empty[DenseVector[Double], Set[DenseVector[Double]]]
    clusterMeans.foreach(m => initialClusters += (m -> Set.empty[DenseVector[Double]]))

    val clusters = dataPoints.foldLeft(initialClusters) { (acc, dp) =>

      val nearestMean = clusterMeans.foldLeft((Double.MaxValue, DenseVector.zeros[Double](dataDim))) { (acc, mean) =>
        val meanDist = distance(dp, mean)
        if (meanDist < acc._1) (meanDist, mean) else acc
      }._2

      acc + (nearestMean -> (acc(nearestMean)+dp))
    }

    clusters.toSeq.map(cl => Cluster(cl._1, cl._2.toSeq))
  }


  def cluster(dataset: Seq[DenseVector[Double]],
              numClusters: Int,
              distanceFunc: (DenseVector[Double], DenseVector[Double]) => Double): Seq[Cluster] = {

    assert(dataset.size > 0)

    var clusters = initializeClusters(dataset, numClusters)

    var oldClusterMeans = clusters.map(_.mean)
    var newClusterMeans = oldClusterMeans.map(mean => mean.map(_ + 1.0))

    var iterations = 0

    while (oldClusterMeans != newClusterMeans) {

      oldClusterMeans = newClusterMeans
      newClusterMeans = clusters.map(c => computeMean(c.assignedDataPoints))
      clusters = assignDataPoints(newClusterMeans, dataset, distanceFunc)

      iterations += 1
      println(s"iteration ${iterations}")
    }

    clusters
  }

}
