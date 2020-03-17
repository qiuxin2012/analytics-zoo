package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.utils.ThreadPool
import jep.{JepConfig, NamingConventionClassEnquirer, SharedInterpreter}

import scala.reflect.ClassTag

object PythonInterpreter {

  private val thread = new ThreadPool(1)
  private val parThread = Array(0).par
  def getSharedInterpreter(): SharedInterpreter = {
    if (sharedInterpreter == null) {
      createInterpreter()
    }
    sharedInterpreter
  }
  private var sharedInterpreter: SharedInterpreter = createInterpreter()
  private def createInterpreter(): SharedInterpreter = {
    val createInterp = () =>
      try {
        val config: JepConfig = new JepConfig()
        config.setClassEnquirer(new NamingConventionClassEnquirer())
        SharedInterpreter.setConfig(config)
        println("Create jep on thread: " + Thread.currentThread())
        val sharedInterpreter = new SharedInterpreter()
        sharedInterpreter
      } catch {
        case e: Exception =>
          println(e)
          throw e
      }
    if (sharedInterpreter == null) {
      synchronized{
        if (sharedInterpreter == null) {
          sharedInterpreter = threadExecute(createInterp)
          val str =
            s"""
               |import tensorflow as tf
               |tf.compat.v1.set_random_seed(1000)
               |import os
               |""".stripMargin
          exec(str)
        }
      }
    }
    sharedInterpreter
  }

  private def threadExecute[T: ClassTag](task: () => T): T = {
//    thread.invokeAndWait(Array(0).map(i => task
//    ))(0)
    val result = parThread.map { i =>
      task()
    }
    result.apply(0)
  }

  def exec(s: String): Unit = {
    val func = () => {
      println("jep exec on thread: " + Thread.currentThread())
      sharedInterpreter.exec(s)
    }
    threadExecute(func)
  }

  def set(s: String, o: AnyRef): Unit = {
    val func = () => {
      println("jep set on thread: " + Thread.currentThread())
      sharedInterpreter.set(s, o)
    }
    threadExecute(func)
  }

  def getValue[T](name: String): T = {
    val func = () => {
      println("jep getValue on thread: " + Thread.currentThread())
      sharedInterpreter.getValue(name)
    }
    threadExecute(func).asInstanceOf[T]
  }
}
