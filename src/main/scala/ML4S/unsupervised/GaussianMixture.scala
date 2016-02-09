package ML4S.unsupervised

import java.awt.{Color, Paint}
import java.util.Random

import breeze.linalg.{DenseMatrix,DenseVector,sum,*,Axis,argmax}
import breeze.plot._
import breeze.stats.distributions.MultivariateGaussian
import breeze.stats.mean


class GaussianMixture(dataPoints: DenseMatrix[Double],
                      numClusters: Int) {

  val dataDim = dataPoints.cols

  def matrixVertTile(vec: DenseMatrix[Double], repSize: Int): DenseMatrix[Double] = {

    var tiledMat = vec

    (0 to repSize).foreach { rep =>
      tiledMat = DenseMatrix.vertcat(tiledMat, vec)
    }

    tiledMat
  }

  def empiricalCov(data: DenseMatrix[Double]): DenseMatrix[Double] = {

    // println(data)
    val empMean: DenseMatrix[Double] = mean(data(::, *))
    //println("empirical mean: " + empMean.toDenseVector)

    var covariance = DenseMatrix.zeros[Double](dataDim, dataDim)

    (0 to dataPoints.rows - 1).foreach { dpId =>

      val dp = dataPoints(dpId, ::).t

      val dpMinusMu = dp - empMean.toDenseVector

      covariance += dpMinusMu * dpMinusMu.t
    }

    covariance.map(_ / (data.rows - 1))
  }

  def meanUpdate(posteriorMat: DenseMatrix[Double]): Vector[DenseVector[Double]] = {

    //println("posterior matrix: " + posteriorMat)
    (0 to numClusters - 1).foldLeft(Vector.empty[DenseVector[Double]]) { (acc, clustId) =>

      val clustPosterior = posteriorMat(::, clustId)

      val unnormalizedMu = sum(dataPoints(::, *) :* clustPosterior, Axis._0)

      val normalizer = sum(clustPosterior)

      val normalizedMu = unnormalizedMu.map(_ / normalizer)

      acc :+ normalizedMu.toDenseVector
    }

  }


  def covUpdate(posteriorMat: DenseMatrix[Double],
                clusters: Seq[MultivariateGaussian]): Vector[DenseMatrix[Double]] = {

    (0 to numClusters - 1).foldLeft(Vector.empty[DenseMatrix[Double]]) { (acc, clustId) =>

      //posteriors of datapoints for this cluster
      val clustPosterior = posteriorMat(::, clustId)

      //the mean for this cluster
      val mu = clusters(clustId).mean

      //(x_i - mean)(x_i-mean)^T for each datapoint i
      val unscaledCovariances = dataPoints(*, ::).map { dp =>
        (dp - mu) * (dp - mu).t
      }

      var covariance = DenseMatrix.zeros[Double](dataDim, dataDim)

      (0 to dataPoints.rows - 1).foreach { dp =>

        covariance += unscaledCovariances(dp) * clustPosterior(dp)

      }

      //Finally we normalize over the posteriors
      val normalizer = sum(clustPosterior)

      val normalizedCovariance = covariance.map(_ / normalizer)

      acc :+ normalizedCovariance
    }

  }


  def piUpdate(posteriorMat: DenseMatrix[Double]): Vector[Double] = {

    (0 to numClusters - 1).foldLeft(Vector.empty[Double]) { (acc, clustId) =>

      //posteriors of datapoints for this cluster
      val clustPosterior = posteriorMat(::, clustId)

      val newPi = sum(clustPosterior) / dataPoints.rows

      acc :+ newPi
    }

  }


  def eStep(clusters: Seq[MultivariateGaussian],
            pi: Seq[Double]): DenseMatrix[Double] = {


    val clusterProbMat = dataPoints(*, ::).map { dp =>
      val dpProbPerCluster = clusters.map(cluster => cluster.pdf(dp))
      DenseVector(dpProbPerCluster.toArray)
    }

    val priorTiled = matrixVertTile(DenseMatrix(pi), dataPoints.rows - 2)

    val unnormalizedPosterior = clusterProbMat :* priorTiled

    unnormalizedPosterior(*, ::).map { post =>

      val normalizer = sum(post)

      post.map(_ / normalizer)
    }
  }


  def mStep(posteriorMat: DenseMatrix[Double],
            clusters: Seq[MultivariateGaussian]): (Seq[MultivariateGaussian], Vector[Double]) = {


    val newMean = meanUpdate(posteriorMat)

    val newCovarianceMat = covUpdate(posteriorMat, clusters)

    val newPi = piUpdate(posteriorMat)

    val newClusters =
      clusters
        .zipWithIndex
        .map { case (clusterDist, idx) =>

        MultivariateGaussian(mean = newMean(idx),
          covariance = newCovarianceMat(idx))
      }

    (newClusters, newPi)
  }


  def cluster() = {

    val randGen = new Random()

    //Initialize all cluster distributions.
    //All covariances set to empirical covariances.
    //means to random data points.
    val initialCov = empiricalCov(dataPoints)


    var currentClusters = (1 to numClusters).map { clust =>
      val meanId = randGen.nextInt(dataPoints.rows)
      val initialMean = dataPoints(meanId, ::).t

      MultivariateGaussian(
        mean = initialMean,
        covariance = initialCov
      )
    }.toList


    //Also initialize Pi randomly
    var currentPi = {
      val unnormalizedRand = (1 to numClusters).map { clust =>
        randGen.nextInt(100)
      }

      val normalizer = unnormalizedRand.sum.toDouble

      unnormalizedRand.map(_ / normalizer)
    }


    var posteriorUpdated = DenseMatrix.zeros[Double](dataPoints.rows, dataDim)

    val f = Figure()
    f.subplot(0).xlabel = "X-coordinate"
    f.subplot(0).ylabel = "Y-coordinate"
    f.subplot(0).title = "311 Service Noise Complaints"

    val id2Color: Int => Paint = id => id match {
      case 0 => Color.YELLOW
      case 1 => Color.RED
      case 2 => Color.GREEN
      case 3 => Color.BLUE
      case 4 => Color.GRAY
      case _ => Color.BLACK
    }


    for (i <- (0 to 100)) {

      val lastPi = currentPi

      posteriorUpdated = eStep(
        clusters = currentClusters,
        pi = currentPi
      )

      println(currentPi)


      val (clusterUpdated, piUpdated) = mStep(
        posteriorMat = posteriorUpdated,
        clusters = currentClusters
      )

      println(piUpdated)


      currentClusters = clusterUpdated.toList
      currentPi = piUpdated

      val piChange = currentPi
        .zip(lastPi)
        .map(el => math.abs(el._1 - el._2))
        .sum / numClusters

      println("change in pi: " + piChange)
    }



    val argmaxPosterior = posteriorUpdated(*, ::).map { postDist =>
      argmax(postDist)
    }.toArray


    val clustersAndPoints = argmaxPosterior
      .zipWithIndex
      .map { case (clustIdx, dpIdx) =>

      (clustIdx, dataPoints(dpIdx, ::).t)

    }.groupBy(_._1)


    for (cl <- clustersAndPoints) {


      val x = cl._2.map(_._2(0))
      val y = cl._2.map(_._2(1))


      f.subplot(0) += scatter(x, y, { (_: Int) => 1000}, { (_: Int) => id2Color(cl._1)})


    }










 /*   for (cl <- currentClusters.zipWithIndex) {


      val samples = cl._1.sample(1000)

      f.subplot(0) += scatter(samples.map(_(0)), samples.map(_(1)), { (_: Int) => 1000}, { (_: Int) => id2Color(cl._2)})


    }*/


  }
}
