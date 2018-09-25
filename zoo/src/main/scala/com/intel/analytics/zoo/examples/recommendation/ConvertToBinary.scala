package com.intel.analytics.zoo.examples.recommendation

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Executors, ThreadFactory}

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.io.Source

object ConvertToBinary {

  def main(args: Array[String]): Unit = {
//    val loadPytorch = NeuralCFexample.loadPytorchTrain("/tmp/0.txt")
//    convertPytorchTrain("/tmp/0.txt", "/tmp/minibatch")
//    load("/tmp/minibatch")
//    convertPytorchToBinary("/tmp/0.txt", "/tmp/minibatch-b")
//    load2("/tmp/minibatch-b", 10)
//    convertPytorchToBinary(s"/tmp/0.txt", s"/tmp/123/0")
//    loadTrainBinary("/tmp/123/0")
//    (0 to 6).foreach{i =>
//      convertPytorchToBinary(s"$i.txt", s"/mnt/local_disk/bigdl-minibatch/$i")
//    }
    convertPytorchtestTobinary("test-ratings.csv", "test-negative.csv",
    "/tmp/bigdl-minibatch/test")
    loadTestBinary("/tmp/bigdl-minibatch/test")
  }
  def convertPytorchTrain(path: String, des: String, batchSize: Int = 2048): Unit = {
    val startTime = System.currentTimeMillis()
    val lines = Source.fromFile(path).getLines()
//    val os = new DataOutputStream(new FileOutputStream(des))

    var count = 1
    while(lines.hasNext) {
      var i = 1
      val input = Tensor[Float](batchSize, 2)
      val target  = Tensor[Float](batchSize, 1)
      while(i <= batchSize && lines.hasNext) {
        val line = lines.next().split(",").map(_.toFloat)
        input.setValue(i, 1, line(0) + 1)
        input.setValue(i, 2, line(1) + 1)
        target.setValue(i, 1, line(2))
        i += 1
      }
      val miniBatch = if (i <= batchSize) {
        input.resize(i - 1, 2)
        target.resize(i - 1, 1)
        MiniBatch(input, target)
      } else {
        MiniBatch(input, target)
      }

      File.save(miniBatch, s"${des}/${count}.obj")
      count += 1
    }
    println(s"save path: ${System.currentTimeMillis() - startTime}ms")
  }

  def load(path: String): Array[MiniBatch[Float]] = {
    val startTime = System.currentTimeMillis()
    var i = 1
    val dir = new java.io.File(path)
    val numFile = dir.listFiles().filter(_.getName().endsWith("obj")).size
    val miniBatches = new Array[MiniBatch[Float]](numFile)
    while(i <= numFile) {
      miniBatches(i - 1) = File.load[MiniBatch[Float]](s"${path}/${i}.obj")
      i += 1
    }
    println(s"load time: ${System.currentTimeMillis() - startTime}ms")
    miniBatches
  }

  def convertPytorchtestTobinary(posFile: String, negFile: String,
                                 des: String, numNegs: Int = 999): Unit = {
    val startTime = System.currentTimeMillis()
    val positives = Source.fromFile(posFile).getLines()
    val negatives = Source.fromFile(negFile).getLines()
    val testByteBuffer = ByteBuffer.allocate((numNegs + 2) * 4)
    val testBuffer = testByteBuffer.asFloatBuffer()
    (1 to numNegs + 1).foreach{i =>
      testBuffer.put(i, 1)
    }

    new java.io.File(des).mkdirs()
    while(positives.hasNext && negatives.hasNext) {
      val pos = positives.next().split("\t")
      val userId = pos(0).toFloat
      val testFile = new DataOutputStream(new FileOutputStream(des + s"/${userId.toInt + 1}.obj"))
      val posItem = pos(1).toFloat
      val neg = negatives.next().split("\t").map(_.toFloat)
      val distinctNegs = neg.distinct
      testBuffer.put(0, distinctNegs.length + 1)

      var i = 1
      while (i <= distinctNegs.length) {
        testBuffer.put(i, distinctNegs(i - 1) + 1)
        i += 1
      }
      testBuffer.put(i, posItem + 1)

      testFile.write(testByteBuffer.array())
      testFile.close()

    }
    println(s"convert test path: ${System.currentTimeMillis() - startTime}ms")
  }

  def convertPytorchToBinary(path: String, des: String, batchSize: Int = 2048): Unit = {
    val startTime = System.currentTimeMillis()
    val lines = Source.fromFile(path).getLines()
    val labelBuffer = ByteBuffer.allocate(batchSize)
    val inputBuffer = ByteBuffer.allocate(batchSize * 4 * 2)
    new java.io.File(des + s"/label").mkdirs()
    new java.io.File(des + s"/input").mkdirs()

    var count = 1
    while(lines.hasNext) {

      val labelFile = new DataOutputStream(new FileOutputStream(des + s"/label/${count}.obj"))
      val inputFile = new DataOutputStream(new FileOutputStream(des + s"/input/${count}.obj"))

      var i = 0
      while(i < batchSize && lines.hasNext) {
        val line = lines.next().split(",").map(_.toFloat)
        inputBuffer.putFloat(2 * i * 4, line(0) + 1)
        inputBuffer.putFloat(2 * i * 4 + 4, line(1) + 1)
        labelBuffer.put(i, line(2).toByte)
        i += 1
      }
      if (i < batchSize - 1) {
        inputFile.write(inputBuffer.array().slice(0, 2 * i * 4))
        labelFile.write(labelBuffer.array().slice(0, i))
        inputFile.close()
        labelFile.close()
      } else{
        inputFile.write(inputBuffer.array())
        labelFile.write(labelBuffer.array())
        inputFile.close()
        labelFile.close()
      }

      count += 1
    }
    println(s"save path: ${System.currentTimeMillis() - startTime}ms")
  }
    val context = new ExecutionContext {
        val threadPool = Executors.newFixedThreadPool(10, new ThreadFactory {
          override def newThread(r: Runnable): Thread = {
            val t = Executors.defaultThreadFactory().newThread(r)
            t.setDaemon(true)
            t
          }
        })

        def execute(runnable: Runnable) {
          threadPool.submit(runnable)
        }

        def reportFailure(t: Throwable) {}
      }

  def loadTrainBinary(path: String, processes: Int = 10): Array[MiniBatch[Float]] = {
    println(s"load from $path")
    val startTime = System.currentTimeMillis()

    val inputDir = path + "/input"
    val labelDir = path + "/label"
    val dir = new java.io.File(inputDir)
    val numFile = dir.listFiles().filter(_.getName().endsWith("obj")).size
    val miniBatches = new Array[MiniBatch[Float]](numFile)
    (1 to numFile).map(i => Future {
        try {
      val inputFileName = inputDir + s"/${i}.obj"
      val labelFileName = labelDir + s"/${i}.obj"
      val inputBytes = Files.readAllBytes(Paths.get(inputFileName))
      val labelBytes = Files.readAllBytes(Paths.get(labelFileName))
      val label = Tensor[Float](labelBytes.map(_.toFloat), Array(labelBytes.size, 1))
      val input = Tensor[Float](labelBytes.size, 2)
      ByteBuffer.wrap(inputBytes).asFloatBuffer.get(input.storage().array())
      miniBatches(i - 1) = MiniBatch(input, label)
        } catch {
          case t : Throwable =>
//            logger.error("Error: " + ExceptionUtils.getStackTrace(t))
            throw t
        }
    }(context)).map(future => {
      Await.result(future, Duration.Inf)
    })

    println(s"load time: ${System.currentTimeMillis() - startTime}ms")
    miniBatches
  }

  def loadTestBinary(testDir: String, processes: Int = 10): Array[Sample[Float]] = {
    println(s"load from $testDir")
    val startTime = System.currentTimeMillis()

    val dir = new java.io.File(testDir)
    val numFile = dir.listFiles().filter(_.getName().endsWith("obj")).size
    val samples = new Array[Sample[Float]](numFile)
    (1 to numFile).map(i => Future {
      try {
        val testFileName = testDir + s"/${i}.obj"
        val testBytes = Files.readAllBytes(Paths.get(testFileName))
        val size = (testBytes.size / 4) - 1
        val label = Tensor[Float](Array(size, 1))
        val input = Tensor[Float](size, 2).fill(i)
        val buffer = ByteBuffer.wrap(testBytes).asFloatBuffer

        (1 to size).foreach{index =>
          input.setValue(index, 2, buffer.get(index))
        }

        label.setValue(buffer.get(0).toInt, 1, 1)
        samples(i - 1) = Sample(input, label)
      } catch {
        case t : Throwable =>
          //            logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          throw t
      }
    }(context)).map(future => {
      Await.result(future, Duration.Inf)
    })

    println(s"load time: ${System.currentTimeMillis() - startTime}ms")
    samples
  }
}
