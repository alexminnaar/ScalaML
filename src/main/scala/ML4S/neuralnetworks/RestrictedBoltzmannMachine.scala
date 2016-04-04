package ML4S.neuralnetworks

import breeze.linalg.{sum, DenseMatrix, DenseVector, Axis, *}
import breeze.numerics.sigmoid
import breeze.stats.distributions.Gaussian
import scala.util.Random
import com.quantifind.charts.Highcharts._

object RestrictedBoltzmannMachine {

  //TODO: Allow for multiple iterations of contrastive divergence
  def contrastiveDivergence(visibles: DenseMatrix[Double],
                            weights: DenseMatrix[Double],
                            visibleBias: DenseMatrix[Double],
                            hiddenBias: DenseMatrix[Double]):
  (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], Double) = {

    val hiddenSample1: DenseMatrix[Double] = weights * visibles
    val hiddenSample2: DenseMatrix[Double] = hiddenSample1(::, *) + hiddenBias.toDenseVector
    val hiddenProbs = sigmoid(hiddenSample2)

    val binaryHidden = binarize(hiddenProbs)

    val positive: DenseMatrix[Double] = hiddenProbs * visibles.t

    val resampleVisible1: DenseMatrix[Double] = weights.t * binaryHidden
    val resampleVisible2: DenseMatrix[Double] = resampleVisible1(::, *) + visibleBias.toDenseVector

    val resampleVisible3 = sigmoid(resampleVisible2)

    val resampleHidden1: DenseMatrix[Double] = weights * resampleVisible3
    val resampleHidden2: DenseMatrix[Double] = sum(resampleVisible3, Axis._1) * hiddenBias
    val resampleHidden3: DenseMatrix[Double] = resampleHidden1 + resampleHidden2.t
    val binaryResampleHiddenProbs = sigmoid(resampleHidden3)
    val binaryResampleHidden = binarize(binaryResampleHiddenProbs)

    val negative: DenseMatrix[Double] = binaryResampleHiddenProbs * resampleVisible3.t

    val recontructionError = sum((visibles - resampleVisible3).map(x => math.pow(x, 2)))

    val weightGrad = positive - negative
    val vBiasGrad = sum((visibles - resampleVisible3), Axis._1).asDenseMatrix
    val hBiasGrad = sum((hiddenProbs - binaryResampleHiddenProbs), Axis._1).asDenseMatrix

    (
      weightGrad,
      vBiasGrad,
      hBiasGrad,
      recontructionError
      )

  }


  //binarize hidden units by setting them to 1 if greater than a random uniform sample or 0 if less.
  def binarize(h: DenseMatrix[Double]): DenseMatrix[Double] = {

    val randGen = new Random()
    h.map(hUnit => if (hUnit > randGen.nextDouble()) 1.0 else 0.0)
  }

  //squash a sequence of vectors horizontally into a DenseMatrix
  def squashVectors(vectSeq: Seq[DenseVector[Double]]): DenseMatrix[Double] = {

    val squashMat = DenseMatrix.zeros[Double](vectSeq.size, vectSeq.head.length)

    vectSeq.zipWithIndex.foreach { case (v, idx) =>
      squashMat(idx, ::) := v.t
    }

    squashMat
  }


  def learn(data: Seq[DenseVector[Double]],
            numHiddenUnits: Int,
            batchSize: Int,
            initialLearningRate: Double) = {

    val numVisibleUnits = data.head.length

    //intialize weights and biases
    var weights = new DenseMatrix[Double](
      numHiddenUnits,
      numVisibleUnits,
      Gaussian.distribution(0, 0.01).sample(numVisibleUnits * numHiddenUnits).toArray
    )

    var lastWUpdate = DenseMatrix.zeros[Double](
      numHiddenUnits,
      numVisibleUnits
    )

    var hiddenBiases = DenseMatrix.zeros[Double](1, numHiddenUnits)
    var visibleBiases = DenseMatrix.zeros[Double](1, numVisibleUnits)

    //Set initial previous update to zero for momentum
    var lastHBUpdate = DenseMatrix.zeros[Double](1, numHiddenUnits)
    var lastVBUpdate = DenseMatrix.zeros[Double](1, numVisibleUnits)


    var learningRate = initialLearningRate

    var errorTracker = Vector.empty[Double]

    var numEpochs = 0

    while (numEpochs < 500) {

      val momentum = numEpochs match {
        case n if n > 5 => 0.9
        case _ => 0.5
      }

      println(momentum)
      learningRate = initialLearningRate / (1.0 + numEpochs / 500.0)

      var epochReconError = 0.0

      val minibatches = data.grouped(batchSize)

      while (minibatches.hasNext) {

        val minibatch = squashVectors(minibatches.next()).t


        val (weightGradient, vBiasGradient, hBiasGradient, reconError) = contrastiveDivergence(
          minibatch,
          weights,
          visibleBiases,
          hiddenBiases
        )


        epochReconError += reconError

        val weightUpdate = momentum * lastWUpdate + (learningRate / batchSize) * weightGradient
        val vBiasUpdate = momentum * lastVBUpdate + (learningRate / batchSize) * vBiasGradient
        val hBiasUpdate = momentum * lastHBUpdate + (learningRate / batchSize) * hBiasGradient

        //keep track of last update for momentum
        lastWUpdate = weightUpdate
        lastVBUpdate = vBiasUpdate
        lastHBUpdate = hBiasUpdate

        weights = weights + weightUpdate
        visibleBiases = visibleBiases + vBiasUpdate
        hiddenBiases = hiddenBiases + hBiasUpdate

      }

      numEpochs += 1

      errorTracker :+= epochReconError

    }

    line(0 to errorTracker.length - 1, errorTracker)
    title("Restricted Boltzmann Machine Reconstruction Error")
    xAxis("Epochs")
    yAxis("Reconstruction Error")

  }

}