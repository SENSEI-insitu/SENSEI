/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnicodeString.cxx

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkUnicodeString.h"

#include "svtkObject.h"
#include <svtk_utf8.h>

#include <map>
#include <stdexcept>

///////////////////////////////////////////////////////////////////////////
// svtkUnicodeString::const_iterator

svtkUnicodeString::const_iterator::const_iterator() = default;

svtkUnicodeString::const_iterator::const_iterator(std::string::const_iterator position)
  : Position(position)
{
}

svtkUnicodeString::value_type svtkUnicodeString::const_iterator::operator*() const
{
  return utf8::unchecked::peek_next(this->Position);
}

bool svtkUnicodeString::const_iterator::operator==(const const_iterator& rhs) const
{
  return this->Position == rhs.Position;
}

bool svtkUnicodeString::const_iterator::operator!=(const const_iterator& rhs) const
{
  return !(*this == rhs);
}

svtkUnicodeString::const_iterator& svtkUnicodeString::const_iterator::operator++()
{
  utf8::unchecked::next(this->Position);
  return *this;
}

svtkUnicodeString::const_iterator svtkUnicodeString::const_iterator::operator++(int)
{
  const_iterator result(this->Position);
  utf8::unchecked::next(this->Position);
  return result;
}

svtkUnicodeString::const_iterator& svtkUnicodeString::const_iterator::operator--()
{
  utf8::unchecked::prior(this->Position);
  return *this;
}

svtkUnicodeString::const_iterator svtkUnicodeString::const_iterator::operator--(int)
{
  const_iterator result(this->Position);
  utf8::unchecked::prior(this->Position);
  return result;
}

///////////////////////////////////////////////////////////////////////////
// svtkUnicodeString::back_insert_iterator

// We provide our own implementation of std::back_insert_iterator for
// use with MSVC 6, where push_back() isn't implemented for std::string.

class svtkUnicodeString::back_insert_iterator
{
public:
  back_insert_iterator(std::string& container)
    : Container(&container)
  {
  }

  back_insert_iterator& operator*() { return *this; }

  back_insert_iterator& operator++() { return *this; }

  back_insert_iterator& operator++(int) { return *this; }

  back_insert_iterator& operator=(std::string::const_reference value)
  {
    this->Container->push_back(value);
    return *this;
  }

private:
  std::string* Container;
};

///////////////////////////////////////////////////////////////////////////
// svtkUnicodeString

svtkUnicodeString::svtkUnicodeString() = default;

svtkUnicodeString::svtkUnicodeString(const svtkUnicodeString& rhs)
  : Storage(rhs.Storage)
{
}

svtkUnicodeString::svtkUnicodeString(size_type count, value_type character)
{
  for (size_type i = 0; i != count; ++i)
    utf8::append(character, svtkUnicodeString::back_insert_iterator(this->Storage));
}

svtkUnicodeString::svtkUnicodeString(const_iterator first, const_iterator last)
  : Storage(first.Position, last.Position)
{
}

bool svtkUnicodeString::is_utf8(const char* value)
{
  return svtkUnicodeString::is_utf8(std::string(value ? value : ""));
}

bool svtkUnicodeString::is_utf8(const std::string& value)
{
  return utf8::is_valid(value.begin(), value.end());
}

svtkUnicodeString svtkUnicodeString::from_utf8(const char* value)
{
  return svtkUnicodeString::from_utf8(std::string(value ? value : ""));
}

svtkUnicodeString svtkUnicodeString::from_utf8(const char* begin, const char* end)
{
  svtkUnicodeString result;
  if (utf8::is_valid(begin, end))
  {
    result.Storage = std::string(begin, end);
  }
  else
  {
    svtkGenericWarningMacro("svtkUnicodeString::from_utf8(): not a valid UTF-8 string.");
  }
  return result;
}

svtkUnicodeString svtkUnicodeString::from_utf8(const std::string& value)
{
  svtkUnicodeString result;
  if (utf8::is_valid(value.begin(), value.end()))
  {
    result.Storage = value;
  }
  else
  {
    svtkGenericWarningMacro("svtkUnicodeString::from_utf8(): not a valid UTF-8 string.");
  }
  return result;
}

svtkUnicodeString svtkUnicodeString::from_utf16(const svtkTypeUInt16* value)
{
  svtkUnicodeString result;

  if (value)
  {
    size_type length = 0;
    while (value[length])
      ++length;

    try
    {
      utf8::utf16to8(value, value + length, svtkUnicodeString::back_insert_iterator(result.Storage));
    }
    catch (utf8::invalid_utf16&)
    {
      svtkGenericWarningMacro(<< "svtkUnicodeString::from_utf16(): not a valid UTF-16 string.");
    }
  }

  return result;
}

svtkUnicodeString& svtkUnicodeString::operator=(const svtkUnicodeString& rhs)
{
  if (this == &rhs)
    return *this;

  this->Storage = rhs.Storage;
  return *this;
}

svtkUnicodeString::const_iterator svtkUnicodeString::begin() const
{
  return const_iterator(this->Storage.begin());
}

svtkUnicodeString::const_iterator svtkUnicodeString::end() const
{
  return const_iterator(this->Storage.end());
}

svtkUnicodeString::value_type svtkUnicodeString::at(size_type offset) const
{
  if (offset >= this->character_count())
    throw std::out_of_range("character out-of-range");

  std::string::const_iterator iterator = this->Storage.begin();
  utf8::unchecked::advance(iterator, offset);
  return utf8::unchecked::peek_next(iterator);
}

svtkUnicodeString::value_type svtkUnicodeString::operator[](size_type offset) const
{
  std::string::const_iterator iterator = this->Storage.begin();
  utf8::unchecked::advance(iterator, offset);
  return utf8::unchecked::peek_next(iterator);
}

const char* svtkUnicodeString::utf8_str() const
{
  return this->Storage.c_str();
}

void svtkUnicodeString::utf8_str(std::string& result) const
{
  result = this->Storage;
}

std::vector<svtkTypeUInt16> svtkUnicodeString::utf16_str() const
{
  std::vector<svtkTypeUInt16> result;
  utf8::unchecked::utf8to16(this->Storage.begin(), this->Storage.end(), std::back_inserter(result));
  return result;
}

void svtkUnicodeString::utf16_str(std::vector<svtkTypeUInt16>& result) const
{
  result.clear();
  utf8::unchecked::utf8to16(this->Storage.begin(), this->Storage.end(), std::back_inserter(result));
}

svtkUnicodeString::size_type svtkUnicodeString::byte_count() const
{
  return this->Storage.size();
}

svtkUnicodeString::size_type svtkUnicodeString::character_count() const
{
  return utf8::unchecked::distance(this->Storage.begin(), this->Storage.end());
}

bool svtkUnicodeString::empty() const
{
  return this->Storage.empty();
}

const svtkUnicodeString::size_type svtkUnicodeString::npos = std::string::npos;

svtkUnicodeString& svtkUnicodeString::operator+=(value_type value)
{
  this->push_back(value);
  return *this;
}

svtkUnicodeString& svtkUnicodeString::operator+=(const svtkUnicodeString& rhs)
{
  this->append(rhs);
  return *this;
}

void svtkUnicodeString::push_back(value_type character)
{
  try
  {
    utf8::append(character, svtkUnicodeString::back_insert_iterator(this->Storage));
  }
  catch (utf8::invalid_code_point&)
  {
    svtkGenericWarningMacro(
      "svtkUnicodeString::push_back(): " << character << "is not a valid Unicode code point");
  }
}

void svtkUnicodeString::append(const svtkUnicodeString& value)
{
  this->Storage.append(value.Storage);
}

void svtkUnicodeString::append(size_type count, value_type character)
{
  try
  {
    this->Storage.append(svtkUnicodeString(count, character).Storage);
  }
  catch (utf8::invalid_code_point&)
  {
    svtkGenericWarningMacro(
      "svtkUnicodeString::append(): " << character << "is not a valid Unicode code point");
  }
}

void svtkUnicodeString::append(const_iterator first, const_iterator last)
{
  this->Storage.append(first.Position, last.Position);
}

void svtkUnicodeString::assign(const svtkUnicodeString& value)
{
  this->Storage.assign(value.Storage);
}

void svtkUnicodeString::assign(size_type count, value_type character)
{
  try
  {
    this->Storage.assign(svtkUnicodeString(count, character).Storage);
  }
  catch (utf8::invalid_code_point&)
  {
    svtkGenericWarningMacro(
      "svtkUnicodeString::assign(): " << character << "is not a valid Unicode code point");
  }
}

void svtkUnicodeString::assign(const_iterator first, const_iterator last)
{
  this->Storage.assign(first.Position, last.Position);
}

void svtkUnicodeString::clear()
{
  this->Storage.clear();
}

svtkUnicodeString svtkUnicodeString::fold_case() const
{
  typedef std::map<value_type, svtkUnicodeString> map_t;

  static map_t map;
  if (map.empty())
  {
#include "svtkUnicodeCaseFoldData.h"

    for (value_type* i = &svtkUnicodeCaseFoldData[0]; *i; ++i)
    {
      const value_type code = *i;
      svtkUnicodeString mapping;
      for (++i; *i; ++i)
      {
        mapping.push_back(*i);
      }
      map.insert(std::make_pair(code, mapping));
    }
  }

  svtkUnicodeString result;

  for (svtkUnicodeString::const_iterator source = this->begin(); source != this->end(); ++source)
  {
    map_t::const_iterator target = map.find(*source);
    if (target != map.end())
    {
      result.append(target->second);
    }
    else
    {
      result.push_back(*source);
    }
  }

  return result;
}

int svtkUnicodeString::compare(const svtkUnicodeString& rhs) const
{
  return this->Storage.compare(rhs.Storage);
}

svtkUnicodeString svtkUnicodeString::substr(size_type offset, size_type count) const
{
  std::string::const_iterator from = this->Storage.begin();
  std::string::const_iterator last = this->Storage.end();

  while (from != last && offset--)
    utf8::unchecked::advance(from, 1);

  std::string::const_iterator to = from;
  while (to != last && count--)
    utf8::unchecked::advance(to, 1);

  return svtkUnicodeString(from, to);
}

void svtkUnicodeString::swap(svtkUnicodeString& rhs)
{
  std::swap(this->Storage, rhs.Storage);
}

bool operator==(const svtkUnicodeString& lhs, const svtkUnicodeString& rhs)
{
  return lhs.compare(rhs) == 0;
}

bool operator!=(const svtkUnicodeString& lhs, const svtkUnicodeString& rhs)
{
  return lhs.compare(rhs) != 0;
}

bool operator<(const svtkUnicodeString& lhs, const svtkUnicodeString& rhs)
{
  return lhs.compare(rhs) < 0;
}

bool operator<=(const svtkUnicodeString& lhs, const svtkUnicodeString& rhs)
{
  return lhs.compare(rhs) <= 0;
}

bool operator>=(const svtkUnicodeString& lhs, const svtkUnicodeString& rhs)
{
  return lhs.compare(rhs) >= 0;
}

bool operator>(const svtkUnicodeString& lhs, const svtkUnicodeString& rhs)
{
  return lhs.compare(rhs) > 0;
}
