package ML4s.neuralnetworks

import ML4S.neuralnetworks.FeedForwardNN
import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid
import org.scalatest.{Matchers, WordSpec}


class FeedForwardNNSpec extends WordSpec with Matchers {

  "FeedForwardNN" should {

    "initialize weight matrices of the correct shape" in {


      val layerDim = Seq(4, 3, 2, 1)

      val weights = FeedForwardNN.initializeWeights(layerDim)

      weights.foreach(w => println(w.rows, w.cols))


    }



    "compute the correct forward pass" in {

      /*

      val layerDim = Seq(4, 3, 2, 1)

      val weights = FeedForwardNN.initializeWeights(layerDim)

      val act: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)

      val input = DenseMatrix(1.0, 2.0, 2.0, 1.0)

      val out = FeedForwardNN.forwardPass(input, weights, act)

      out._1.foreach(x => println(x))

      println(out._2)

      */

      val act: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)


      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      val input = DenseMatrix(0.0, 0.0)

      val out = FeedForwardNN.forwardPass(input, Seq(w12.t,w23.t), act)

      println("result")
      out._1.foreach(x => println(x))

      println(out._2)

      out._2 should equal(DenseMatrix(0.3676098854895219))


    }

    "compute the correct backward pass" in {

/*
      val layerDim = Seq(4, 3, 2, 1)

      val weights = FeedForwardNN.initializeWeights(layerDim)

      val actFn: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)

      val input = DenseMatrix(1.0, 2.0, 2.0, 1.0)

      val (act, pred) = FeedForwardNN.forwardPass(input, weights, actFn)

      val actFnDerivative = (x: DenseMatrix[Double]) => x.map(y => sigmoid(y) * (1 - sigmoid(y)))

      val tar = DenseMatrix(1.0)

      val grad = FeedForwardNN.backwardPass(act, weights, pred, tar, actFn, actFnDerivative)


      println("gradients")
      grad.foreach(x => println(x.rows, x.cols))

      */

      val actFn: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)
      val actFnDerivative = (x: DenseMatrix[Double]) => x.map(y => sigmoid(y) * (1 - sigmoid(y)))


      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      val input = DenseMatrix(0.0, 0.0)

      val out = FeedForwardNN.forwardPass(input, Seq(w12.t,w23.t), actFn)

      println("activations")
      out._1.foreach(println)
      val tar = DenseMatrix(0.0)

      val grad = FeedForwardNN.backwardPass(out._1, Seq(w12.t,w23.t), out._2, tar, actFn, actFnDerivative)

      println("Final Grad")

      grad.foreach(println)




    }

    "compute the correct sgd updates" in {

      /*
      val layerDim = Seq(4, 3, 2, 1)

      val weights = FeedForwardNN.initializeWeights(layerDim)

      val actFn: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)

      val input = DenseMatrix(1.0, 2.0, 2.0, 1.0)

      val (act, pred) = FeedForwardNN.forwardPass(input, weights, actFn)

      val actFnDerivative = (x: DenseMatrix[Double]) => x.map(y => sigmoid(y) * (1 - sigmoid(y)))

      val tar = DenseMatrix(1.0)

      val grad = FeedForwardNN.backwardPass(act, weights, pred, tar, actFn, actFnDerivative)


      weights.foreach(println)
      val newWeights = FeedForwardNN.sgd(weights, grad, 0.1)
      newWeights.foreach(println)
      */

      val actFn: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)
      val actFnDerivative = (x: DenseMatrix[Double]) => x.map(y => sigmoid(y) * (1 - sigmoid(y)))


      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      val input = DenseMatrix(0.0, 0.0)

      val out = FeedForwardNN.forwardPass(input, Seq(w12.t,w23.t), actFn)

      //println("activations")
      //out._1.foreach(println)
      val tar = DenseMatrix(0.0)

      val grad = FeedForwardNN.backwardPass(out._1, Seq(w12.t,w23.t), out._2, tar, actFn, actFnDerivative)

      println("gradients")
      grad.foreach(println)

      val newWeights = FeedForwardNN.sgd(Seq(w12.t,w23.t), grad, 0.2)

      println("new weights")
      println(newWeights)


    }



    "solve the xor problem" in {

      val actFn: DenseMatrix[Double] => DenseMatrix[Double] = (x: DenseMatrix[Double]) => sigmoid(x)
      val actFnDerivative = (x: DenseMatrix[Double]) => x.map(y => sigmoid(y) * (1 - sigmoid(y)))

      val xor = Vector((DenseMatrix( 0.0, 0.0), DenseMatrix(0.0)),
        (DenseMatrix( 1.0, 0.0), DenseMatrix(1.0)),
        (DenseMatrix( 0.0, 1.0), DenseMatrix(1.0)),
        (DenseMatrix( 1.0, 1.0), DenseMatrix(0.0)))

      val w12 = DenseMatrix((0.341232, 0.129952, -0.923123),
        (-0.115223, 0.570345, -0.328932))

      val w23 = DenseMatrix((-0.993423, 0.164732, 0.752621))

      var weights = Seq(w12.t, w23.t)


      val stepSize = 0.2
      val numIter = 5000

      //train network using xor examples
      for (i <- (0 to numIter)) {


        for (ex <- xor) {

          val (activations,output) = FeedForwardNN.forwardPass(ex._1,weights,actFn)
          val der = FeedForwardNN.backwardPass(activations, weights,output,ex._2,actFn,actFnDerivative)

          weights = FeedForwardNN.sgd(weights, der, stepSize)
        }

      }

      //test xor function
      val (o1, a1) = FeedForwardNN.forwardPass(xor(0)._1,weights, actFn)
      println(a1)
      //a1 should equal(DenseMatrix(0.06558841610114152))


      val (o2, a2) = FeedForwardNN.forwardPass(xor(1)._1,weights, actFn)
      println(a2)
      //a2 should equal(DenseMatrix(0.9420490971682289))

      val (o3, a3) = FeedForwardNN.forwardPass( xor(2)._1,weights,actFn)
      println(a3)
      //a3 should equal(DenseMatrix(0.9236073162109176))

      val (o4, a4) = FeedForwardNN.forwardPass(xor(3)._1,weights, actFn)
      println(a4)
      //a4 should equal(DenseMatrix(0.056917630053021126))

    }




  }


}
