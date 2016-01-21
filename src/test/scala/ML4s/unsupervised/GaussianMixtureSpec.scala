package ML4s.unsupervised

import ML4S.unsupervised.GaussianMixture
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.{Matchers, WordSpec}


class GaussianMixtureSpec extends WordSpec with Matchers {

  "The Gaussian Mixture Model" should {


    val testData = DenseMatrix(
      (0.1, 0.2),
      (0.3, -0.1),
      (-0.2, 0.1),
      (-0.1, 0.1)
    )


    val testClusters = Seq(

      MultivariateGaussian(
        mean = DenseVector(0.0, 0.0),
        covariance = DenseMatrix((1.0, -0.9), (-0.9, 1.0)))
      ,
      MultivariateGaussian(
        mean = DenseVector(0.2, 0.1),
        covariance = DenseMatrix((1.0, 0.5), (0.5, 1.0))
      ),
      MultivariateGaussian(
        mean = DenseVector(-0.1, 0.4),
        covariance = DenseMatrix((0.5, -0.4), (-0.4, 0.5))
      )

    )

    val testPriors = Seq(0.1, 0.3, 0.6)



    val gmm = new GaussianMixture(
      dataPoints = testData,
      numClusters = 3
    )


    val empCov=gmm.empiricalCov(testData)

    println("empirical covariance: "+empCov)

      "compute the correct E-Step" in {


      val posterior = gmm.eStep(
        clusters = testClusters,
        pi = testPriors
      )

      println("posterior: "+posterior)

    }


    "compute the correct mean update in the M-Step" in {

      val posterior = gmm.eStep(
        clusters = testClusters,
        pi = testPriors
      )


      val meanUpdate = gmm.meanUpdate(
        posteriorMat = posterior
      )

      println(meanUpdate)
    }



    "compute the correct covariance update in the M-Step" in {

      val posterior = gmm.eStep(
        clusters = testClusters,
        pi = testPriors
      )

      val covUpdate = gmm.covUpdate(
        posteriorMat = posterior,
        clusters = testClusters
      )

      println(covUpdate)

      val blah=MultivariateGaussian(DenseVector(0.0, 0.0),covUpdate(0))

      println(blah)

    }

  }

}
