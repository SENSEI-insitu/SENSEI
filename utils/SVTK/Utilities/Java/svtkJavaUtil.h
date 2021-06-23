/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkJavaUtil.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef svtkJavaUtil_h
#define svtkJavaUtil_h

#include "svtkCommand.h"
#include "svtkJavaModule.h"
#include "svtkSystemIncludes.h"
#include <jni.h>

#include <string>

extern SVTKJAVA_EXPORT jlong q(JNIEnv* env, jobject obj);

extern SVTKJAVA_EXPORT void* svtkJavaGetPointerFromObject(JNIEnv* env, jobject obj);
extern SVTKJAVA_EXPORT char* svtkJavaUTFToChar(JNIEnv* env, jstring in);
extern SVTKJAVA_EXPORT bool svtkJavaUTFToString(JNIEnv* env, jstring in, std::string& out);
extern SVTKJAVA_EXPORT jstring svtkJavaMakeJavaString(JNIEnv* env, const char* in);

extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfFloatFromFloat(
  JNIEnv* env, const float* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfDoubleFromFloat(
  JNIEnv* env, const float* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfDoubleFromDouble(
  JNIEnv* env, const double* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfShortFromShort(
  JNIEnv* env, const short* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfIntFromInt(JNIEnv* env, const int* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfIntFromIdType(
  JNIEnv* env, const svtkIdType* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfIntFromLongLong(
  JNIEnv* env, const long long* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfIntFromSignedChar(
  JNIEnv* env, const signed char* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfLongFromLong(
  JNIEnv* env, const long* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfByteFromUnsignedChar(
  JNIEnv* env, const unsigned char* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfByteFromChar(
  JNIEnv* env, const char* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfCharFromChar(
  JNIEnv* env, const char* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfUnsignedCharFromUnsignedChar(
  JNIEnv* env, const unsigned char* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfUnsignedIntFromUnsignedInt(
  JNIEnv* env, const unsigned int* arr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfUnsignedShortFromUnsignedShort(
  JNIEnv* env, const unsigned short* ptr, int size);
extern SVTKJAVA_EXPORT jarray svtkJavaMakeJArrayOfUnsignedLongFromUnsignedLong(
  JNIEnv* env, const unsigned long* arr, int size);

// this is the void pointer parameter passed to the svtk callback routines on
// behalf of the Java interface for callbacks.
struct svtkJavaVoidFuncArg
{
  JavaVM* vm;
  jobject uobj;
  jmethodID mid;
};

extern SVTKJAVA_EXPORT void svtkJavaVoidFunc(void*);
extern SVTKJAVA_EXPORT void svtkJavaVoidFuncArgDelete(void*);

class SVTKJAVA_EXPORT svtkJavaCommand : public svtkCommand
{
public:
  static svtkJavaCommand* New() { return new svtkJavaCommand; }

  void SetGlobalRef(jobject obj) { this->uobj = obj; }
  void SetMethodID(jmethodID id) { this->mid = id; }
  void AssignJavaVM(JNIEnv* env) { env->GetJavaVM(&(this->vm)); }

  void Execute(svtkObject*, unsigned long, void*);

  JavaVM* vm;
  jobject uobj;
  jmethodID mid;

protected:
  svtkJavaCommand();
  ~svtkJavaCommand();
};

#endif
// SVTK-HeaderTest-Exclude: svtkJavaUtil.h
