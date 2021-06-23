/*=========================================================================

  Program:   ParaView
  Module:    svtkSelection.h

  Copyright (c) Kitware, Inc.
  All rights reserved.
  See Copyright.txt or http://www.paraview.org/HTML/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class svtkSelection
 * @brief data object that represents a "selection" in SVTK.
 *
 * svtkSelection is a data object that represents a selection definition. It is
 * used to define the elements that are selected. The criteria of the selection
 * is defined using one or more svtkSelectionNode instances. Parameters of the
 * svtkSelectionNode define what kind of elements are being selected
 * (svtkSelectionNode::GetFieldType), how the selection criteria is defined
 * (svtkSelectionNode::GetContentType), etc.
 *
 * Filters like svtkExtractSelection, svtkExtractDataArraysOverTime can be used to
 * extract the selected elements from a dataset.
 *
 * @section CombiningSelection Combining Selections
 *
 * When a svtkSelection contains multiple svtkSelectionNode instances, the
 * selection defined is a union of all the elements identified by each of the
 * nodes.
 *
 * Optionally, one can use `svtkSelection::SetExpression` to define a boolean
 * expression to build arbitrarily complex combinations. The expression can be
 * defined using names assigned to the selection nodes when the nodes are added
 * to svtkSelection (either explicitly or automatically).
 *
 * @sa
 * svtkSelectionNode
 */

#ifndef svtkSelection_h
#define svtkSelection_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"
#include "svtkSmartPointer.h" // for  svtkSmartPointer.

#include <memory> // for unique_ptr.
#include <string> // for string.

class svtkSelectionNode;
class svtkSignedCharArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkSelection : public svtkDataObject
{
public:
  svtkTypeMacro(svtkSelection, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkSelection* New();

  /**
   * Restore data object to initial state,
   */
  void Initialize() override;

  /**
   * Returns SVTK_SELECTION enumeration value.
   */
  int GetDataObjectType() override { return SVTK_SELECTION; }

  /**
   * Returns the number of nodes in this selection.
   * Each node contains information about part of the selection.
   */
  unsigned int GetNumberOfNodes() const;

  /**
   * Returns a node given it's index. Performs bound checking
   * and will return nullptr if out-of-bounds.
   */
  virtual svtkSelectionNode* GetNode(unsigned int idx) const;

  /**
   * Returns a node with the given name, if present, else nullptr is returned.
   */
  virtual svtkSelectionNode* GetNode(const std::string& name) const;

  /**
   * Adds a selection node. Assigns the node a unique name and returns that
   * name. This API is primarily provided for backwards compatibility and
   * `SetNode` is the preferred method.
   */
  virtual std::string AddNode(svtkSelectionNode*);

  /**
   * Adds a svtkSelectionNode and assigns it the specified name. The name
   * must be a non-empty string. If an item with the same name
   * has already been added, it will be removed.
   */
  virtual void SetNode(const std::string& name, svtkSelectionNode*);

  /**
   * Returns the name for a node at the given index.
   */
  virtual std::string GetNodeNameAtIndex(unsigned int idx) const;

  //@{
  /**
   * Removes a selection node.
   */
  virtual void RemoveNode(unsigned int idx);
  virtual void RemoveNode(const std::string& name);
  virtual void RemoveNode(svtkSelectionNode*);
  //@}

  /**
   * Removes all selection nodes.
   */
  virtual void RemoveAllNodes();

  //@{
  /**
   * Get/Set the expression that defines the boolean expression to combine the
   * selection nodes. Expression consists of node name identifiers, `|` for
   * boolean-or, '&' for boolean and, '!' for boolean not, and parenthesis `(`
   * and `)`. If the expression consists of a node name identifier that is not
   * assigned any `svtkSelectionNode` (using `SetNode`) then it is evaluates to
   * `false`.
   *
   * `SetExpression` does not validate the expression. It will be validated in
   * `Evaluate` call.
   */
  svtkSetMacro(Expression, std::string);
  svtkGetMacro(Expression, std::string);
  //@}

  /**
   * Copy selection nodes of the input.
   */
  void DeepCopy(svtkDataObject* src) override;

  /**
   * Copy selection nodes of the input.
   * This is a shallow copy: selection lists and pointers in the
   * properties are passed by reference.
   */
  void ShallowCopy(svtkDataObject* src) override;

  /**
   * Union this selection with the specified selection.
   * Attempts to reuse selection nodes in this selection if properties
   * match exactly. Otherwise, creates new selection nodes.
   */
  virtual void Union(svtkSelection* selection);

  /**
   * Union this selection with the specified selection node.
   * Attempts to reuse a selection node in this selection if properties
   * match exactly. Otherwise, creates a new selection node.
   */
  virtual void Union(svtkSelectionNode* node);

  /**
   * Remove the nodes from the specified selection from this selection.
   * Assumes that selection node internal arrays are svtkIdTypeArrays.
   */
  virtual void Subtract(svtkSelection* selection);

  /**
   * Remove the nodes from the specified selection from this selection.
   * Assumes that selection node internal arrays are svtkIdTypeArrays.
   */
  virtual void Subtract(svtkSelectionNode* node);

  /**
   * Return the MTime taking into account changes to the properties
   */
  svtkMTimeType GetMTime() override;

  //@{
  /**
   * Dumps the contents of the selection, giving basic information only.
   */
  virtual void Dump();
  virtual void Dump(ostream& os);
  //@}

  //@{
  /**
   * Retrieve a svtkSelection stored inside an invormation object.
   */
  static svtkSelection* GetData(svtkInformation* info);
  static svtkSelection* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Evaluates the expression for each element in the values. The order
   * matches the order of the selection nodes. If not expression is set or if
   * it's an empty string, then an expression that simply combines all selection
   * nodes in an binary-or is assumed.
   */
  svtkSmartPointer<svtkSignedCharArray> Evaluate(
    svtkSignedCharArray* const* values, unsigned int num_values) const;

  /**
   * Convenience method to pass a map of svtkSignedCharArray ptrs (or
   * svtkSmartPointers).
   */
  template <typename MapType>
  svtkSmartPointer<svtkSignedCharArray> Evaluate(const MapType& values_map) const;

protected:
  svtkSelection();
  ~svtkSelection() override;

  std::string Expression;

private:
  svtkSelection(const svtkSelection&) = delete;
  void operator=(const svtkSelection&) = delete;

  class svtkInternals;
  svtkInternals* Internals;
};

//----------------------------------------------------------------------------
template <typename MapType>
inline svtkSmartPointer<svtkSignedCharArray> svtkSelection::Evaluate(const MapType& values_map) const
{
  const unsigned int num_nodes = this->GetNumberOfNodes();
  std::unique_ptr<svtkSignedCharArray*[]> values(new svtkSignedCharArray*[num_nodes]);
  for (unsigned int cc = 0; cc < num_nodes; ++cc)
  {
    auto iter = values_map.find(this->GetNodeNameAtIndex(cc));
    values[cc] = iter != values_map.end() ? iter->second : nullptr;
  }
  return this->Evaluate(&values[0], num_nodes);
}

#endif
