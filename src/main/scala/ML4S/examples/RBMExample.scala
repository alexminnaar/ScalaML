package ML4S.examples

import ML4S.neuralnetworks.RestrictedBoltzmannMachine
import breeze.linalg.DenseVector

object RBMExample extends App {

  val dataset = Seq(
    DenseVector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    DenseVector(1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    DenseVector(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    DenseVector(0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
    DenseVector(0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
    DenseVector(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
  )


  RestrictedBoltzmannMachine.learn(data = dataset,
    numHiddenUnits = 2,
    batchSize = 6,
    initialLearningRate = 0.05)


}