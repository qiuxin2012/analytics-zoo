/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_analytics_zoo_pipeline_api_net_PytorchModel */

#ifndef _Included_com_intel_analytics_zoo_pipeline_api_net_PytorchModel
#define _Included_com_intel_analytics_zoo_pipeline_api_net_PytorchModel
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    loadNative
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadNative
  (JNIEnv *, jobject, jstring);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    forwardNative
 * Signature: (J[FI[I)Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_forwardNative
  (JNIEnv *, jobject, jlong, jfloatArray, jint, jintArray);

#ifdef __cplusplus
}
#endif
#endif
