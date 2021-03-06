/*=========================================================================

  Program:   Visualization Toolkit
  Module:    otherByteSwap.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// this program tests the byte swapper

#include "svtkByteSwap.h"
#include "svtkDebugLeaks.h"

#include <sstream>

int TestByteSwap(ostream& strm)
{
  // actual test
  strm << "Test svtkByteSwap Start" << endl;

  char check[1024];
  short sword[2];
  char cword[1024];
  unsigned short usword[2];

  memcpy(sword, "abcd", 2);
  svtkByteSwap::Swap2BE(sword);
  memcpy(check, sword, 2);
  strm << "Swap2BE(short \"ab\") -> " << check[0] << check[1] << endl;

  memcpy(usword, "abcd", 2);
  svtkByteSwap::Swap2BE(usword);
  memcpy(check, usword, 2);
  strm << "Swap2BE(unsigned short \"ab\") -> " << check[0] << check[1] << endl;

  memcpy(cword, "abcd", 4);
  svtkByteSwap::Swap4BE(cword);
  memcpy(check, cword, 4);
  strm << "Swap4BE(char *\"abcd\") -> " << check[0] << check[1] << check[2] << check[3] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap8BE(cword);
  memcpy(check, cword, 8);
  strm << "Swap8BE(char *\"abcdefgh\") -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap2BERange(cword, 4);
  memcpy(check, cword, 8);
  strm << "Swap2BERange(char *\"abcdefgh\",4) -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap4BERange(cword, 2);
  memcpy(check, cword, 8);
  strm << "Swap4BERange(char *\"abcdefgh\",2) -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap8BERange(cword, 1);
  memcpy(check, cword, 8);
  strm << "Swap8BERange(char *\"abcdefgh\",1) -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefghijklmnop", 16);
  strm << "SwapWrite2BERange(char *\"abcdefghijklmnop\",8,stdout) -> ";
  svtkByteSwap::SwapWrite2BERange(cword, 8, stdout);
  strm << endl;

  memcpy(cword, "abcdefghijklmnop", 16);
  strm << "SwapWrite4BERange(char *\"abcdefghijklmnop\",4,stdout) -> ";
  svtkByteSwap::SwapWrite4BERange(cword, 4, stdout);
  strm << endl;

  memcpy(cword, "abcdefghijklmnop", 16);
  strm << "SwapWrite8BERange(char *\"abcdefghijklmnop\",2,stdout) -> ";
  svtkByteSwap::SwapWrite8BERange(cword, 2, stdout);
  strm << endl;

  memcpy(cword, "abcdefghijklmnop", 16);
  strm << "SwapWrite2BERange(char *\"abcdefghijklmnop\",8,&strm) -> ";
  svtkByteSwap::SwapWrite2BERange(cword, 8, &strm);
  strm << endl;

  memcpy(cword, "abcdefghijklmnop", 16);
  strm << "SwapWrite4BERange(char *\"abcdefghijklmnop\",4,&strm) -> ";
  svtkByteSwap::SwapWrite4BERange(cword, 4, &strm);
  strm << endl;

  memcpy(cword, "abcdefghijklmnop", 16);
  strm << "SwapWrite8BERange(char *\"abcdefghijklmnop\",2,&strm) -> ";
  svtkByteSwap::SwapWrite8BERange(cword, 2, &strm);
  strm << endl;

  memcpy(sword, "abcd", 2);
  svtkByteSwap::Swap2LE(sword);
  memcpy(check, sword, 2);
  strm << "Swap2LE(short \"ab\") -> " << check[0] << check[1] << endl;

  memcpy(usword, "abcd", 2);
  svtkByteSwap::Swap2LE(usword);
  memcpy(check, usword, 2);
  strm << "Swap2LE(unsigned short \"ab\") -> " << check[0] << check[1] << endl;

  memcpy(cword, "abcd", 4);
  svtkByteSwap::Swap4LE(cword);
  memcpy(check, cword, 4);
  strm << "Swap4LE(char *\"abcd\") -> " << check[0] << check[1] << check[2] << check[3] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap8LE(cword);
  memcpy(check, cword, 8);
  strm << "Swap8LE(char *\"abcdefgh\") -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap2LERange(cword, 4);
  memcpy(check, cword, 8);
  strm << "Swap2LERange(char *\"abcdefgh\",4) -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap4LERange(cword, 2);
  memcpy(check, cword, 8);
  strm << "Swap4LERange(char *\"abcdefgh\",2) -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::Swap8LERange(cword, 1);
  memcpy(check, cword, 8);
  strm << "Swap8LERange(char *\"abcdefgh\",1) -> " << check[0] << check[1] << check[2] << check[3]
       << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::SwapVoidRange(cword, 4, 2);
  memcpy(check, cword, 8);
  strm << "SwapVoidRange(char *\"abcdefgh\",4,2) -> " << check[0] << check[1] << check[2]
       << check[3] << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::SwapVoidRange(cword, 2, 4);
  memcpy(check, cword, 8);
  strm << "SwapVoidRange(char *\"abcdefgh\",2,4) -> " << check[0] << check[1] << check[2]
       << check[3] << check[4] << check[5] << check[6] << check[7] << endl;

  memcpy(cword, "abcdefgh", 8);
  svtkByteSwap::SwapVoidRange(cword, 1, 8);
  memcpy(check, cword, 8);
  strm << "SwapVoidRange(char *\"abcdefgh\",1,8) -> " << check[0] << check[1] << check[2]
       << check[3] << check[4] << check[5] << check[6] << check[7] << endl;

  strm << "Test svtkByteSwap End" << endl;
  return 0;
}

int otherByteSwap(int, char*[])
{
  std::ostringstream svtkmsg_with_warning_C4701;
  return TestByteSwap(svtkmsg_with_warning_C4701);
}
