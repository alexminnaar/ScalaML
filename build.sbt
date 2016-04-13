name := "ML4S"

version := "1.0"

scalaVersion := "2.11.7"


libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2",
  "com.quantifind" %% "wisp" % "0.0.4",
  "org.scala-saddle" %% "saddle-core" % "1.3.4",
  "com.typesafe.akka" %% "akka-actor" % "2.3.12",
  "org.scalatest" %% "scalatest" % "2.2.5",
  "org.apache.spark" %% "spark-core" % "1.6.1",
  "org.apache.spark" %% "spark-mllib" % "1.6.1"
)