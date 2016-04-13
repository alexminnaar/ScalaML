package ML4S.akka

import java.util.UUID

import ML4S.akka.NNSlave.TopK
import akka.actor.{Props, ActorRef, Actor, ActorLogging}
import breeze.linalg.DenseVector

object NNMaster {

  case class QueryInput(input: DenseVector[Double])

  case class Prediction(p: String)

}


class NNMaster(inputs: Seq[DenseVector[Double]],
               outputs: Seq[String],
               k: Int,
               distanceFn: (DenseVector[Double], DenseVector[Double]) => Double,
               numPartitions: Int) extends Actor with ActorLogging {

  import NNMaster._

  //partition data
  val partitionedInput = inputs
    .grouped(inputs.size / numPartitions)
    .toSeq

  val partitionedOutput = outputs
    .grouped(outputs.size / numPartitions)
    .toSeq

  log.info(s"data partitioned into ${partitionedInput.size} chunks")

  //create actors to handle each partition
  val partitionActors: Array[ActorRef] = new Array[ActorRef](partitionedInput.size)

  partitionedInput.zipWithIndex.foreach { case (inputPartition, idx) =>

    val outputParition = partitionedOutput(idx)

    partitionActors(idx) = context.actorOf(Props(new NNSlave(
      UUID.randomUUID(),
      inputPartition,
      outputParition,
      k,
      distanceFn
    )))

  }

  log.info("Slave actors created")


  var slavesNotFinished = numPartitions
  var mergedDistances = Vector.empty[(String, Double)]

  def receive = {

    case QueryInput(input) => {
      partitionActors.foreach(_ ! QueryInput(input))

      context.become(waitForSlaves)
    }


  }


  def waitForSlaves: Receive = {

    case TopK(id, nn) => {

      log.info(s"slave ${id} search results received by master")

      slavesNotFinished -= 1

      log.info(s"${slavesNotFinished} workers still working...")
      mergedDistances ++= nn

      if (slavesNotFinished == 0) {

        log.info("All results computed")

        val overallTopK = mergedDistances
          .sortBy(_._2)
          .take(k)

        //Most frequent class in top K
        val pred = overallTopK
          .groupBy(identity)
          .mapValues(_.size)
          .maxBy(_._2)._1

        log.info(s"Prediction is: ${pred._1}")

        context.parent ! Prediction(pred._1)

        context.unbecome()
      }
    }


  }


}
