package com.intel.analytics.zoo.common

import java.util.concurrent.{ExecutorService, Executors, ThreadFactory}

import com.intel.analytics.bigdl.utils.ThreadPool
import com.intel.analytics.zoo.common.PythonInterpreter.sharedInterpreter
import jep.{JepConfig, NamingConventionClassEnquirer, SharedInterpreter}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger

import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag

object PythonInterpreter {
  private val logger = Logger.getLogger(this.getClass)
  private var threadPool: ExecutorService = null

  private val context = new ExecutionContext {
    threadPool = Executors.newFixedThreadPool(1, new ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = Executors.defaultThreadFactory().newThread(r)
        t.setName("default-thread-computing " + t.getId)
        t.setDaemon(true)
        t
      }
    })

    def execute(runnable: Runnable) {
      threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable): Unit = {
      throw t
    }
  }
  //  private val parThread = Array(0).par
  def getSharedInterpreter(): SharedInterpreter = {
    if (sharedInterpreter == null) {
      this.synchronized{
        createInterpreter()
      }
    }
    sharedInterpreter
  }
  private var sharedInterpreter: SharedInterpreter = createInterpreter()
  private def createInterpreter(): SharedInterpreter = {
    val createInterp = () => {
      println("Create jep on thread: " + Thread.currentThread())
      val config: JepConfig = new JepConfig()
        config.setClassEnquirer(new NamingConventionClassEnquirer())
        SharedInterpreter.setConfig(config)
        val sharedInterpreter = new SharedInterpreter()
        sharedInterpreter
      }
    if (sharedInterpreter == null) {
      sharedInterpreter = threadExecute(createInterp)
      val str =
        s"""
           |import tensorflow as tf
           |tf.compat.v1.set_random_seed(1000)
           |import os
           |""".stripMargin
      //          exec(str)
      //      }
    }
    sharedInterpreter
  }

  private def threadExecute[T](task: () => T, timeout: Duration = Duration.Inf): T = {
    try {
      val re = Array(task).map(t => Future {
        t()
      }(context)).map(future => {
        Await.result(future, timeout)
      })
      re(0)
    } catch {
      case t : Throwable =>
        logger.error("Error: " + ExceptionUtils.getStackTrace(t))
        throw t
    }
  }

  def exec(s: String): Unit = {
    val func = () => {
      println(s"jep exec ${s} on thread: " + Thread.currentThread())
      sharedInterpreter.exec(s)
      println("jep exec finished")
    }
    threadExecute(func)
  }

  def set(s: String, o: AnyRef): Unit = {
    val func = () => {
      println(s"jep set ${s} on thread: " + Thread.currentThread())
      sharedInterpreter.set(s, o)
      println("jep set finished")
    }
    threadExecute(func)
  }

  def getValue[T](name: String): T = {
    val func = () => {
      println(s"jep getValue ${name} on thread: " + Thread.currentThread())
      val re = sharedInterpreter.getValue(name)
      println("jep getValue finished")
      re
    }
    threadExecute(func).asInstanceOf[T]
  }
}
