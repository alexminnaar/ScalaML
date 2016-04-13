package ML4S.akka

import java.util.UUID

import ML4S.akka.NNMaster.QueryInput
import akka.actor.{Actor, ActorLogging}
import breeze.linalg.{*, DenseVector}

object NNSlave {

  case class TopK(slaveId: UUID, neighbours: Seq[(String, Double)])

}

class NNSlave(id: UUID,
              inputPartition: Seq[DenseVector[Double]],
              outputPartition: Seq[String],
              k: Int,
              distanceFn: (DenseVector[Double], DenseVector[Double]) => Double) extends Actor with ActorLogging {

  import NNSlave._

  val slaveData = inputPartition

  def receive = {

    case QueryInput(input) => {

      log.info(s"slave ${id} received query")

      //compute similarity for each example.
      val distances = slaveData
        .map(r => distanceFn(r, input))

      //Get top k most similar classes
      val topKClasses = distances
        .toArray
        .zipWithIndex
        .sortBy(_._1)
        .take(k)
        .map { case (dist, idx) => (outputPartition(idx), dist) }


      sender() ! TopK(id, topKClasses)

      log.info(s"slave ${id} finished nearest neighbor search")

    }

  }


}
