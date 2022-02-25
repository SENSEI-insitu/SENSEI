/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkJavaUtil.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifdef _INTEGRAL_MAX_BITS
#undef _INTEGRAL_MAX_BITS
#endif
#define _INTEGRAL_MAX_BITS 64

#include "svtkDebugLeaks.h"
#include "svtkObject.h"
#include "svtkWindows.h"

#include "svtkJavaUtil.h"

// Silence warning like
// "dereferencing type-punned pointer will break strict-aliasing rules"
// it happens because this kind of expression: (void **)(&e)
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

JNIEXPORT jlong svtkJavaGetId(JNIEnv* env, jobject obj)
{
  jfieldID id;
  jlong result;

  id = env->GetFieldID(env->GetObjectClass(obj), "svtkId", "J");

  result = env->GetLongField(obj, id);
  return result;
}

JNIEXPORT void* svtkJavaGetPointerFromObject(JNIEnv* env, jobject obj)
{
  return obj ? (void*)(size_t)svtkJavaGetId(env, obj) : 0;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfDoubleFromDouble(JNIEnv* env, const double* ptr, int size)
{
  jdoubleArray ret;
  int i;
  jdouble* array;

  ret = env->NewDoubleArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetDoubleArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseDoubleArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfDoubleFromFloat(JNIEnv* env, const float* ptr, int size)
{
  jdoubleArray ret;
  int i;
  jdouble* array;

  ret = env->NewDoubleArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetDoubleArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseDoubleArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfIntFromInt(JNIEnv* env, const int* ptr, int size)
{
  jintArray ret;
  int i;
  jint* array;

  ret = env->NewIntArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetIntArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseIntArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfIntFromIdType(JNIEnv* env, const svtkIdType* ptr, int size)
{
  jintArray ret;
  int i;
  jint* array;

  ret = env->NewIntArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetIntArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = (int)ptr[i];
  }

  env->ReleaseIntArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfIntFromLongLong(JNIEnv* env, const long long* ptr, int size)
{
  jintArray ret;
  int i;
  jint* array;

  ret = env->NewIntArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetIntArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = (int)ptr[i];
  }

  env->ReleaseIntArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfIntFromSignedChar(JNIEnv* env, const signed char* ptr, int size)
{
  jintArray ret;
  int i;
  jint* array;

  ret = env->NewIntArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetIntArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = (int)ptr[i];
  }

  env->ReleaseIntArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfFloatFromFloat(JNIEnv* env, const float* ptr, int size)
{
  jfloatArray ret;
  int i;
  jfloat* array;

  ret = env->NewFloatArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetFloatArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseFloatArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfShortFromShort(JNIEnv* env, const short* ptr, int size)
{
  jshortArray ret;
  int i;
  jshort* array;

  ret = env->NewShortArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetShortArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseShortArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfByteFromUnsignedChar(
  JNIEnv* env, const unsigned char* ptr, int size)
{
  jbyteArray ret;
  int i;
  jbyte* array;

  ret = env->NewByteArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetByteArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseByteArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfLongFromLong(JNIEnv* env, const long* ptr, int size)
{
  cout.flush();
  jlongArray ret;
  int i;
  jlong* array;

  ret = env->NewLongArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetLongArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseLongArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfUnsignedLongFromUnsignedLong(
  JNIEnv* env, const unsigned long* ptr, int size)
{
  cout.flush();
  jlongArray ret;
  int i;
  jlong* array;

  ret = env->NewLongArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetLongArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseLongArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfUnsignedShortFromUnsignedShort(
  JNIEnv* env, const unsigned short* ptr, int size)
{
  cout.flush();
  jshortArray ret;
  int i;
  jshort* array;

  ret = env->NewShortArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetShortArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseShortArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfUnsignedCharFromUnsignedChar(
  JNIEnv* env, const unsigned char* ptr, int size)
{
  cout.flush();
  jbyteArray ret;
  int i;
  jbyte* array;

  ret = env->NewByteArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetByteArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseByteArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfUnsignedIntFromUnsignedInt(
  JNIEnv* env, const unsigned int* ptr, int size)
{
  cout.flush();
  jintArray ret;
  int i;
  jint* array;

  ret = env->NewIntArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetIntArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseIntArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfCharFromChar(JNIEnv* env, const char* ptr, int size)
{
  cout.flush();
  jcharArray ret;
  int i;
  jchar* array;

  ret = env->NewCharArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetCharArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseCharArrayElements(ret, array, 0);
  return ret;
}

JNIEXPORT jarray svtkJavaMakeJArrayOfIntFromBool(JNIEnv* env, const bool* ptr, int size)
{
  cout.flush();
  jintArray ret;
  int i;
  jint* array;

  ret = env->NewIntArray(size);
  if (ret == 0)
  {
    // should throw an exception here
    return 0;
  }

  array = env->GetIntArrayElements(ret, nullptr);

  // copy the data
  for (i = 0; i < size; i++)
  {
    array[i] = ptr[i];
  }

  env->ReleaseIntArrayElements(ret, array, 0);
  return ret;
}

// http://java.sun.com/docs/books/jni/html/pitfalls.html#12400
static void JNU_ThrowByName(JNIEnv* env, const char* name, const char* msg)
{
  jclass cls = env->FindClass(name);
  /* if cls is nullptr, an exception has already been thrown */
  if (cls != nullptr)
  {
    env->ThrowNew(cls, msg);
  }
  /* free the local ref */
  env->DeleteLocalRef(cls);
}

static char* JNU_GetStringNativeChars(JNIEnv* env, jstring jstr)
{
  if (jstr == nullptr)
  {
    return nullptr;
  }
  jbyteArray bytes = 0;
  jthrowable exc;
  char* result = 0;
  if (env->EnsureLocalCapacity(2) < 0)
  {
    return 0; /* out of memory error */
  }
  jclass Class_java_lang_String = env->GetObjectClass(jstr);
  jmethodID MID_String_getBytes =
    env->GetMethodID(Class_java_lang_String, "getBytes", "(Ljava/lang/String;)[B");
  jstring encoding = env->NewStringUTF("UTF-8");
  bytes = (jbyteArray)env->CallObjectMethod(jstr, MID_String_getBytes, encoding);
  env->DeleteLocalRef(encoding);
  exc = env->ExceptionOccurred();
  if (!exc)
  {
    jint len = env->GetArrayLength(bytes);
    result = new char[len + 1];

    if (result == 0)
    {
      JNU_ThrowByName(env, "java/lang/OutOfMemoryError", 0);
      env->DeleteLocalRef(bytes);
      return 0;
    }
    env->GetByteArrayRegion(bytes, 0, len, (jbyte*)result);
    result[len] = 0; /* nullptr-terminate */
  }
  else
  {
    env->DeleteLocalRef(exc);
  }
  env->DeleteLocalRef(bytes);
  return result;
}

JNIEXPORT char* svtkJavaUTFToChar(JNIEnv* env, jstring in)
{
  return JNU_GetStringNativeChars(env, in);
}

JNIEXPORT bool svtkJavaUTFToString(JNIEnv* env, jstring in, std::string& out)
{
  const char* cstring = JNU_GetStringNativeChars(env, in);
  if (cstring)
  {
    out = cstring;
    delete[] cstring;
    return true;
  }

  return false;
}

JNIEXPORT jstring svtkJavaMakeJavaString(JNIEnv* env, const char* in)
{
  if (!in)
  {
    return env->NewStringUTF("");
  }
  else
  {
    size_t length = strlen(in);
    jbyteArray bytes = env->NewByteArray((jsize)length);
    env->SetByteArrayRegion(bytes, 0, (jsize)length, (jbyte*)in);

    jclass Class_java_lang_String = env->FindClass("java/lang/String");
    jmethodID MID_String_ctor =
      env->GetMethodID(Class_java_lang_String, "<init>", "([BLjava/lang/String;)V");

    jstring encoding = env->NewStringUTF("UTF-8");
    jstring result =
      (jstring)env->NewObject(Class_java_lang_String, MID_String_ctor, bytes, encoding);
    env->DeleteLocalRef(encoding);
    env->DeleteLocalRef(bytes);

    return result;
  }
}

//**jcp this is the callback interface stub for Java. no user parms are passed
// since the callback must be a method of a class. We make the rash assumption
// that the <this> pointer will anchor any required other elements for the
// called functions. - edited by km
JNIEXPORT void svtkJavaVoidFunc(void* f)
{
  svtkJavaVoidFuncArg* iprm = (svtkJavaVoidFuncArg*)f;
  // make sure we have a valid method ID
  if (iprm->mid)
  {
    JNIEnv* e;
    // it should already be atached
#ifdef JNI_VERSION_1_2
    iprm->vm->AttachCurrentThread((void**)(&e), nullptr);
#else
    iprm->vm->AttachCurrentThread((JNIEnv_**)(&e), nullptr);
#endif
    e->CallVoidMethod(iprm->uobj, iprm->mid, nullptr);
  }
}

JNIEXPORT void svtkJavaVoidFuncArgDelete(void* arg)
{
  svtkJavaVoidFuncArg* arg2;

  arg2 = (svtkJavaVoidFuncArg*)arg;

  JNIEnv* e;
  // it should already be atached
#ifdef JNI_VERSION_1_2
  arg2->vm->AttachCurrentThread((void**)(&e), nullptr);
#else
  arg2->vm->AttachCurrentThread((JNIEnv_**)(&e), nullptr);
#endif
  // free the structure
  e->DeleteGlobalRef(arg2->uobj);
  delete arg2;
}

svtkJavaCommand::svtkJavaCommand()
{
  this->vm = nullptr;
}

svtkJavaCommand::~svtkJavaCommand()
{
  JNIEnv* e;
  // it should already be atached
#ifdef JNI_VERSION_1_2
  this->vm->AttachCurrentThread((void**)(&e), nullptr);
#else
  this->vm->AttachCurrentThread((JNIEnv_**)(&e), nullptr);
#endif
  // free the structure
  e->DeleteGlobalRef(this->uobj);
}

void svtkJavaCommand::Execute(svtkObject*, unsigned long, void*)
{
  // make sure we have a valid method ID
  if (this->mid)
  {
    JNIEnv* e;
    // it should already be atached
#ifdef JNI_VERSION_1_2
    this->vm->AttachCurrentThread((void**)(&e), nullptr);
#else
    this->vm->AttachCurrentThread((JNIEnv_**)(&e), nullptr);
#endif
    e->CallVoidMethod(this->uobj, this->mid, nullptr);
  }
}
