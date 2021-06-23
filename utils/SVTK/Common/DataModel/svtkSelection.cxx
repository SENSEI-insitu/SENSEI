/*=========================================================================

  Program:   ParaView
  Module:    svtkSelection.cxx

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSelection.h"

#include "svtkAbstractArray.h"
#include "svtkFieldData.h"
#include "svtkInformation.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationIterator.h"
#include "svtkInformationObjectBaseKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkSMPTools.h"
#include "svtkSelectionNode.h"
#include "svtkSignedCharArray.h"
#include "svtkTable.h"

#include <svtksys/RegularExpression.hxx>
#include <svtksys/SystemTools.hxx>

#include <atomic>
#include <cassert>
#include <cctype>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{
// since certain compilers don't support std::to_string yet
template <typename T>
std::string convert_to_string(const T& val)
{
  std::ostringstream str;
  str << val;
  return str.str();
}
}

//============================================================================
namespace parser
{
class Node
{
public:
  Node() = default;
  virtual ~Node() = default;
  virtual bool Evaluate(svtkIdType offset) const = 0;
  virtual void Print(ostream& os) const = 0;
};

class NodeVariable : public Node
{
  svtkSignedCharArray* Data;
  std::string Name;

public:
  NodeVariable(svtkSignedCharArray* data, const std::string& name)
    : Data(data)
    , Name(name)
  {
  }
  bool Evaluate(svtkIdType offset) const override
  {
    assert(this->Data == nullptr || this->Data->GetNumberOfValues() > offset);
    return this->Data ? (this->Data->GetValue(offset) != 0) : false;
  }
  void Print(ostream& os) const override { os << this->Name; }
};

class NodeNot : public Node
{
  std::shared_ptr<Node> Child;

public:
  NodeNot(const std::shared_ptr<Node>& node)
    : Child(node)
  {
  }
  bool Evaluate(svtkIdType offset) const override
  {
    assert(this->Child);
    return !this->Child->Evaluate(offset);
  }
  void Print(ostream& os) const override
  {
    os << "!";
    this->Child->Print(os);
  }
};

class NodeAnd : public Node
{
  std::shared_ptr<Node> ChildA;
  std::shared_ptr<Node> ChildB;

public:
  NodeAnd(const std::shared_ptr<Node>& nodeA, const std::shared_ptr<Node>& nodeB)
    : ChildA(nodeA)
    , ChildB(nodeB)
  {
  }
  bool Evaluate(svtkIdType offset) const override
  {
    assert(this->ChildA && this->ChildB);
    return this->ChildA->Evaluate(offset) && this->ChildB->Evaluate(offset);
  }
  void Print(ostream& os) const override
  {
    os << "(";
    this->ChildA->Print(os);
    os << " & ";
    this->ChildB->Print(os);
    os << ")";
  }
};

class NodeOr : public Node
{
  std::shared_ptr<Node> ChildA;
  std::shared_ptr<Node> ChildB;

public:
  NodeOr(const std::shared_ptr<Node>& nodeA, const std::shared_ptr<Node>& nodeB)
    : ChildA(nodeA)
    , ChildB(nodeB)
  {
  }
  bool Evaluate(svtkIdType offset) const override
  {
    assert(this->ChildA && this->ChildB);
    return this->ChildA->Evaluate(offset) || this->ChildB->Evaluate(offset);
  }
  void Print(ostream& os) const override
  {
    os << "(";
    this->ChildA->Print(os);
    os << " | ";
    this->ChildB->Print(os);
    os << ")";
  }
};
} // namespace parser

//============================================================================
class svtkSelection::svtkInternals
{
  // applies the operator on the "top" (aka back) of the op_stack to the
  // variables on the var_stack and pushes the result on the var_stack.
  bool ApplyBack(
    std::vector<char>& op_stack, std::vector<std::shared_ptr<parser::Node> >& var_stack) const
  {
    assert(!op_stack.empty());

    if (op_stack.back() == '!')
    {
      if (var_stack.empty())
      {
        // failed
        return false;
      }
      const auto a = var_stack.back();
      var_stack.pop_back();
      var_stack.push_back(std::make_shared<parser::NodeNot>(a));
      // pop the applied operator.
      op_stack.pop_back();
      return true;
    }
    else if (op_stack.back() == '|' || op_stack.back() == '&')
    {
      if (var_stack.size() < 2)
      {
        // failed!
        return false;
      }

      const auto b = var_stack.back();
      var_stack.pop_back();
      const auto a = var_stack.back();
      var_stack.pop_back();
      if (op_stack.back() == '|')
      {
        var_stack.push_back(std::make_shared<parser::NodeOr>(a, b));
      }
      else
      {
        var_stack.push_back(std::make_shared<parser::NodeAnd>(a, b));
      }
      // pop the applied operator.
      op_stack.pop_back();
      return true;
    }
    return false;
  }

  // higher the value, higher the precedence.
  inline int precedence(char op) const
  {
    switch (op)
    {
      case '|':
        return -15;
      case '&':
        return -14;
      case '!':
        return -3;
      case '(':
      case ')':
        return -1;
      default:
        return -100;
    }
  }

public:
  std::map<std::string, svtkSmartPointer<svtkSelectionNode> > Items;
  svtksys::RegularExpression RegExID;

  svtkInternals()
    : RegExID("^[a-zA-Z0-9]+$")
  {
  }

  std::shared_ptr<parser::Node> BuildExpressionTree(
    const std::string& expression, const std::map<std::string, svtkSignedCharArray*>& values_map)
  {
    // We don't use PEGTL since it does not support all supported compilers viz.
    // VS2013.
    std::string accumated_text;
    accumated_text.reserve(expression.size() + 64);

    std::vector<std::string> parts;
    for (auto ch : expression)
    {
      switch (ch)
      {
        case '(':
        case ')':
        case '|':
        case '&':
        case '!':
          if (!accumated_text.empty())
          {
            parts.push_back(accumated_text);
            accumated_text.clear();
          }
          parts.push_back(std::string(1, ch));
          break;

        default:
          if (std::isalnum(ch))
          {
            accumated_text.push_back(ch);
          }
          break;
      }
    }
    if (!accumated_text.empty())
    {
      parts.push_back(accumated_text);
    }

    std::vector<std::shared_ptr<parser::Node> > var_stack;
    std::vector<char> op_stack;
    for (const auto& term : parts)
    {
      if (term[0] == '(')
      {
        op_stack.push_back(term[0]);
      }
      else if (term[0] == ')')
      {
        // apply operators till we encounter the opening paren.
        while (!op_stack.empty() && op_stack.back() != '(' && this->ApplyBack(op_stack, var_stack))
        {
        }
        if (op_stack.empty())
        {
          // missing opening paren???
          return nullptr;
        }
        assert(op_stack.back() == '(');
        // pop the opening paren.
        op_stack.pop_back();
      }
      else if (term[0] == '&' || term[0] == '|' || term[0] == '!')
      {
        while (!op_stack.empty() && (precedence(term[0]) < precedence(op_stack.back())) &&
          this->ApplyBack(op_stack, var_stack))
        {
        }
        // push the boolean operator on stack to eval later.
        op_stack.push_back(term[0]);
      }
      else
      {
        auto iter = values_map.find(term);
        auto dataptr = iter != values_map.end() ? iter->second : nullptr;
        var_stack.push_back(std::make_shared<parser::NodeVariable>(dataptr, term));
      }
    }

    while (!op_stack.empty() && this->ApplyBack(op_stack, var_stack))
    {
    }
    return (op_stack.empty() && var_stack.size() == 1) ? var_stack.front() : nullptr;
  }
};

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkSelection);

//----------------------------------------------------------------------------
svtkSelection::svtkSelection()
  : Expression()
  , Internals(new svtkSelection::svtkInternals())
{
  this->Information->Set(svtkDataObject::DATA_EXTENT_TYPE(), SVTK_PIECES_EXTENT);
  this->Information->Set(svtkDataObject::DATA_PIECE_NUMBER(), -1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_PIECES(), 1);
  this->Information->Set(svtkDataObject::DATA_NUMBER_OF_GHOST_LEVELS(), 0);
}

//----------------------------------------------------------------------------
svtkSelection::~svtkSelection()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void svtkSelection::Initialize()
{
  this->Superclass::Initialize();
  this->RemoveAllNodes();
  this->Expression.clear();
}

//----------------------------------------------------------------------------
unsigned int svtkSelection::GetNumberOfNodes() const
{
  return static_cast<unsigned int>(this->Internals->Items.size());
}

//----------------------------------------------------------------------------
svtkSelectionNode* svtkSelection::GetNode(unsigned int idx) const
{
  const svtkInternals& internals = (*this->Internals);
  if (static_cast<unsigned int>(internals.Items.size()) > idx)
  {
    auto iter = std::next(internals.Items.begin(), static_cast<int>(idx));
    assert(iter != internals.Items.end());
    return iter->second;
  }
  return nullptr;
}

//----------------------------------------------------------------------------
svtkSelectionNode* svtkSelection::GetNode(const std::string& name) const
{
  const svtkInternals& internals = (*this->Internals);
  auto iter = internals.Items.find(name);
  if (iter != internals.Items.end())
  {
    return iter->second;
  }
  return nullptr;
}

//----------------------------------------------------------------------------
std::string svtkSelection::AddNode(svtkSelectionNode* node)
{
  if (!node)
  {
    return std::string();
  }

  const svtkInternals& internals = (*this->Internals);

  // Make sure that node is not already added
  for (const auto& pair : internals.Items)
  {
    if (pair.second == node)
    {
      return pair.first;
    }
  }

  static std::atomic<uint64_t> counter(0U);
  std::string name = std::string("node") + convert_to_string(++counter);
  while (internals.Items.find(name) != internals.Items.end())
  {
    name = std::string("node") + convert_to_string(++counter);
  }

  this->SetNode(name, node);
  return name;
}

//----------------------------------------------------------------------------
void svtkSelection::SetNode(const std::string& name, svtkSelectionNode* node)
{
  svtkInternals& internals = (*this->Internals);
  if (!node)
  {
    svtkErrorMacro("`node` cannot be null.");
  }
  else if (!internals.RegExID.find(name))
  {
    svtkErrorMacro("`" << name << "` is not in the expected form.");
  }
  else if (internals.Items[name] != node)
  {
    internals.Items[name] = node;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
std::string svtkSelection::GetNodeNameAtIndex(unsigned int idx) const
{
  const svtkInternals& internals = (*this->Internals);
  if (static_cast<unsigned int>(internals.Items.size()) > idx)
  {
    auto iter = std::next(internals.Items.begin(), static_cast<int>(idx));
    assert(iter != internals.Items.end());
    return iter->first;
  }
  return std::string();
}

//----------------------------------------------------------------------------
void svtkSelection::RemoveNode(unsigned int idx)
{
  svtkInternals& internals = (*this->Internals);
  if (static_cast<unsigned int>(internals.Items.size()) > idx)
  {
    auto iter = std::next(internals.Items.begin(), static_cast<int>(idx));
    assert(iter != internals.Items.end());
    internals.Items.erase(iter);
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkSelection::RemoveNode(const std::string& name)
{
  svtkInternals& internals = (*this->Internals);
  if (internals.Items.erase(name) == 1)
  {
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkSelection::RemoveNode(svtkSelectionNode* node)
{
  svtkInternals& internals = (*this->Internals);
  for (auto iter = internals.Items.begin(); iter != internals.Items.end(); ++iter)
  {
    if (iter->second == node)
    {
      internals.Items.erase(iter);
      this->Modified();
      break;
    }
  }
}

//----------------------------------------------------------------------------
void svtkSelection::RemoveAllNodes()
{
  svtkInternals& internals = (*this->Internals);
  if (!internals.Items.empty())
  {
    internals.Items.clear();
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkSelection::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  unsigned int numNodes = this->GetNumberOfNodes();
  os << indent << "Number of nodes: " << numNodes << endl;
  os << indent << "Nodes: " << endl;
  for (unsigned int i = 0; i < numNodes; i++)
  {
    os << indent << "Node #" << i << endl;
    this->GetNode(i)->PrintSelf(os, indent.GetNextIndent());
  }
}

//----------------------------------------------------------------------------
void svtkSelection::ShallowCopy(svtkDataObject* src)
{
  if (auto* ssrc = svtkSelection::SafeDownCast(src))
  {
    this->Expression = ssrc->Expression;
    this->Internals->Items = ssrc->Internals->Items;
    this->Superclass::ShallowCopy(src);
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkSelection::DeepCopy(svtkDataObject* src)
{
  if (auto* ssrc = svtkSelection::SafeDownCast(src))
  {
    this->Expression = ssrc->Expression;

    const auto& srcMap = ssrc->Internals->Items;
    auto& destMap = this->Internals->Items;
    destMap = srcMap;
    for (auto& apair : destMap)
    {
      svtkNew<svtkSelectionNode> clone;
      clone->DeepCopy(apair.second);
      apair.second = clone;
    }
    this->Superclass::DeepCopy(src);
    this->Modified();
  }
}

//----------------------------------------------------------------------------
void svtkSelection::Union(svtkSelection* s)
{
  for (unsigned int n = 0; n < s->GetNumberOfNodes(); ++n)
  {
    this->Union(s->GetNode(n));
  }
}

//----------------------------------------------------------------------------
void svtkSelection::Union(svtkSelectionNode* node)
{
  bool merged = false;
  for (unsigned int tn = 0; tn < this->GetNumberOfNodes(); ++tn)
  {
    svtkSelectionNode* tnode = this->GetNode(tn);
    if (tnode->EqualProperties(node))
    {
      tnode->UnionSelectionList(node);
      merged = true;
      break;
    }
  }
  if (!merged)
  {
    svtkSmartPointer<svtkSelectionNode> clone = svtkSmartPointer<svtkSelectionNode>::New();
    clone->DeepCopy(node);
    this->AddNode(clone);
  }
}

//----------------------------------------------------------------------------
void svtkSelection::Subtract(svtkSelection* s)
{
  for (unsigned int n = 0; n < s->GetNumberOfNodes(); ++n)
  {
    this->Subtract(s->GetNode(n));
  }
}

//----------------------------------------------------------------------------
void svtkSelection::Subtract(svtkSelectionNode* node)
{
  bool subtracted = false;
  for (unsigned int tn = 0; tn < this->GetNumberOfNodes(); ++tn)
  {
    svtkSelectionNode* tnode = this->GetNode(tn);

    if (tnode->EqualProperties(node))
    {
      tnode->SubtractSelectionList(node);
      subtracted = true;
    }
  }
  if (!subtracted)
  {
    svtkErrorMacro("Could not subtract selections");
  }
}

//----------------------------------------------------------------------------
svtkMTimeType svtkSelection::GetMTime()
{
  svtkMTimeType mtime = this->Superclass::GetMTime();
  const svtkInternals& internals = (*this->Internals);
  for (const auto& apair : internals.Items)
  {
    mtime = std::max(mtime, apair.second->GetMTime());
  }
  return mtime;
}

//----------------------------------------------------------------------------
svtkSelection* svtkSelection::GetData(svtkInformation* info)
{
  return info ? svtkSelection::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkSelection* svtkSelection::GetData(svtkInformationVector* v, int i)
{
  return svtkSelection::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
svtkSmartPointer<svtkSignedCharArray> svtkSelection::Evaluate(
  svtkSignedCharArray* const* values, unsigned int num_values) const
{
  std::map<std::string, svtkSignedCharArray*> values_map;

  svtkIdType numVals = -1;
  unsigned int cc = 0;
  const svtkInternals& internals = (*this->Internals);
  for (const auto& apair : internals.Items)
  {
    svtkSignedCharArray* array = cc < num_values ? values[cc] : nullptr;
    if (array == nullptr)
    {
      // lets assume null means false.
    }
    else
    {
      if (array->GetNumberOfComponents() != 1)
      {
        svtkGenericWarningMacro("Only single-component arrays are supported!");
        return nullptr;
      }
      if (numVals != -1 && array->GetNumberOfTuples() != numVals)
      {
        svtkGenericWarningMacro("Mismatched number of tuples.");
        return nullptr;
      }
      numVals = array->GetNumberOfTuples();
    }
    values_map[apair.first] = array;
    cc++;
  }

  std::string expr = this->Expression;
  if (expr.empty())
  {
    bool add_separator = false;
    std::ostringstream stream;
    for (const auto& apair : internals.Items)
    {
      stream << (add_separator ? "|" : "") << apair.first;
      add_separator = true;
    }
    expr = stream.str();
  }

  auto tree = this->Internals->BuildExpressionTree(expr, values_map);
  if (tree && (!values_map.empty()))
  {
    auto result = svtkSmartPointer<svtkSignedCharArray>::New();
    result->SetNumberOfComponents(1);
    result->SetNumberOfTuples(numVals);
    svtkSMPTools::For(0, numVals, [&](svtkIdType start, svtkIdType end) {
      for (svtkIdType idx = start; idx < end; ++idx)
      {
        result->SetTypedComponent(idx, 0, tree->Evaluate(idx));
      }
    });
    return result;
  }
  else if (!tree)
  {
    svtkGenericWarningMacro("Failed to parse expression: " << this->Expression);
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkSelection::Dump()
{
  this->Dump(cout);
}

//----------------------------------------------------------------------------
void svtkSelection::Dump(ostream& os)
{
  svtkSmartPointer<svtkTable> tmpTable = svtkSmartPointer<svtkTable>::New();
  cerr << "==Selection==" << endl;
  for (unsigned int i = 0; i < this->GetNumberOfNodes(); ++i)
  {
    os << "===Node " << i << "===" << endl;
    svtkSelectionNode* node = this->GetNode(i);
    os << "ContentType: ";
    switch (node->GetContentType())
    {
      case svtkSelectionNode::GLOBALIDS:
        os << "GLOBALIDS";
        break;
      case svtkSelectionNode::PEDIGREEIDS:
        os << "PEDIGREEIDS";
        break;
      case svtkSelectionNode::VALUES:
        os << "VALUES";
        break;
      case svtkSelectionNode::INDICES:
        os << "INDICES";
        break;
      case svtkSelectionNode::FRUSTUM:
        os << "FRUSTUM";
        break;
      case svtkSelectionNode::LOCATIONS:
        os << "LOCATIONS";
        break;
      case svtkSelectionNode::THRESHOLDS:
        os << "THRESHOLDS";
        break;
      case svtkSelectionNode::BLOCKS:
        os << "BLOCKS";
        break;
      case svtkSelectionNode::USER:
        os << "USER";
        break;
      default:
        os << "UNKNOWN";
        break;
    }
    os << endl;
    os << "FieldType: ";
    switch (node->GetFieldType())
    {
      case svtkSelectionNode::CELL:
        os << "CELL";
        break;
      case svtkSelectionNode::POINT:
        os << "POINT";
        break;
      case svtkSelectionNode::FIELD:
        os << "FIELD";
        break;
      case svtkSelectionNode::VERTEX:
        os << "VERTEX";
        break;
      case svtkSelectionNode::EDGE:
        os << "EDGE";
        break;
      case svtkSelectionNode::ROW:
        os << "ROW";
        break;
      default:
        os << "UNKNOWN";
        break;
    }
    os << endl;
    if (node->GetSelectionData())
    {
      tmpTable->SetRowData(node->GetSelectionData());
      tmpTable->Dump(10);
    }
  }
}
