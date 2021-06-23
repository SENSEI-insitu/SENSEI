/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFunctionParser.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkFunctionParser
 * @brief   Parse and evaluate a mathematical expression
 *
 * svtkFunctionParser is a class that takes in a mathematical expression as
 * a char string, parses it, and evaluates it at the specified values of
 * the variables in the input string.
 *
 * You can use the "if" operator to create conditional expressions
 * such as if ( test, trueresult, falseresult). These evaluate the boolean
 * valued test expression and then evaluate either the trueresult or the
 * falseresult expression to produce a final (scalar or vector valued) value.
 * "test" may contain <,>,=,|,&, and () and all three subexpressions can
 * evaluate arbitrary function operators (ln, cos, +, if, etc)
 *
 * @par Thanks:
 * Juha Nieminen (juha.nieminen@gmail.com) for relicensing this branch of the
 * function parser code that this class is based upon under the new BSD license
 * so that it could be used in SVTK. Note, the BSD license applies to this
 * version of the function parser only (by permission of the author), and not
 * the original library.
 *
 * @par Thanks:
 * Thomas Dunne (thomas.dunne@iwr.uni-heidelberg.de) for adding code for
 * two-parameter-parsing and a few functions (sign, min, max).
 *
 * @par Thanks:
 * Sid Sydoriak (sxs@lanl.gov) for adding boolean operations and
 * conditional expressions and for fixing a variety of bugs.
 */

#ifndef svtkFunctionParser_h
#define svtkFunctionParser_h

#include "svtkCommonMiscModule.h" // For export macro
#include "svtkObject.h"
#include "svtkTuple.h" // needed for svtkTuple
#include <string>     // needed for string.
#include <vector>     // needed for vector

#define SVTK_PARSER_IMMEDIATE 1
#define SVTK_PARSER_UNARY_MINUS 2
#define SVTK_PARSER_UNARY_PLUS 3

// supported math functions
#define SVTK_PARSER_ADD 4
#define SVTK_PARSER_SUBTRACT 5
#define SVTK_PARSER_MULTIPLY 6
#define SVTK_PARSER_DIVIDE 7
#define SVTK_PARSER_POWER 8
#define SVTK_PARSER_ABSOLUTE_VALUE 9
#define SVTK_PARSER_EXPONENT 10
#define SVTK_PARSER_CEILING 11
#define SVTK_PARSER_FLOOR 12
#define SVTK_PARSER_LOGARITHM 13
#define SVTK_PARSER_LOGARITHME 14
#define SVTK_PARSER_LOGARITHM10 15
#define SVTK_PARSER_SQUARE_ROOT 16
#define SVTK_PARSER_SINE 17
#define SVTK_PARSER_COSINE 18
#define SVTK_PARSER_TANGENT 19
#define SVTK_PARSER_ARCSINE 20
#define SVTK_PARSER_ARCCOSINE 21
#define SVTK_PARSER_ARCTANGENT 22
#define SVTK_PARSER_HYPERBOLIC_SINE 23
#define SVTK_PARSER_HYPERBOLIC_COSINE 24
#define SVTK_PARSER_HYPERBOLIC_TANGENT 25
#define SVTK_PARSER_MIN 26
#define SVTK_PARSER_MAX 27
#define SVTK_PARSER_SIGN 29

// functions involving vectors
#define SVTK_PARSER_CROSS 28
#define SVTK_PARSER_VECTOR_UNARY_MINUS 30
#define SVTK_PARSER_VECTOR_UNARY_PLUS 31
#define SVTK_PARSER_DOT_PRODUCT 32
#define SVTK_PARSER_VECTOR_ADD 33
#define SVTK_PARSER_VECTOR_SUBTRACT 34
#define SVTK_PARSER_SCALAR_TIMES_VECTOR 35
#define SVTK_PARSER_VECTOR_TIMES_SCALAR 36
#define SVTK_PARSER_VECTOR_OVER_SCALAR 37
#define SVTK_PARSER_MAGNITUDE 38
#define SVTK_PARSER_NORMALIZE 39

// constants involving vectors
#define SVTK_PARSER_IHAT 40
#define SVTK_PARSER_JHAT 41
#define SVTK_PARSER_KHAT 42

// code for if(bool, trueval, falseval) resulting in a scalar
#define SVTK_PARSER_IF 43

// code for if(bool, truevec, falsevec) resulting in a vector
#define SVTK_PARSER_VECTOR_IF 44

// codes for boolean expressions
#define SVTK_PARSER_LESS_THAN 45

// codes for boolean expressions
#define SVTK_PARSER_GREATER_THAN 46

// codes for boolean expressions
#define SVTK_PARSER_EQUAL_TO 47

// codes for boolean expressions
#define SVTK_PARSER_AND 48

// codes for boolean expressions
#define SVTK_PARSER_OR 49

// codes for scalar variables come before those for vectors. Do not define
// values for SVTK_PARSER_BEGIN_VARIABLES+1, SVTK_PARSER_BEGIN_VARIABLES+2, ...,
// because they are used to look up variables numbered 1, 2, ...
#define SVTK_PARSER_BEGIN_VARIABLES 50

// the value that is returned as a result if there is an error
#define SVTK_PARSER_ERROR_RESULT SVTK_FLOAT_MAX

class SVTKCOMMONMISC_EXPORT svtkFunctionParser : public svtkObject
{
public:
  static svtkFunctionParser* New();
  svtkTypeMacro(svtkFunctionParser, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return parser's MTime
   */
  svtkMTimeType GetMTime() override;

  //@{
  /**
   * Set/Get input string to evaluate.
   */
  void SetFunction(const char* function);
  svtkGetStringMacro(Function);
  //@}

  /**
   * Check whether the result is a scalar result.  If it isn't, then
   * either the result is a vector or an error has occurred.
   */
  int IsScalarResult();

  /**
   * Check whether the result is a vector result.  If it isn't, then
   * either the result is scalar or an error has occurred.
   */
  int IsVectorResult();

  /**
   * Get a scalar result from evaluating the input function.
   */
  double GetScalarResult();

  //@{
  /**
   * Get a vector result from evaluating the input function.
   */
  double* GetVectorResult() SVTK_SIZEHINT(3);
  void GetVectorResult(double result[3])
  {
    double* r = this->GetVectorResult();
    result[0] = r[0];
    result[1] = r[1];
    result[2] = r[2];
  }
  //@}

  //@{
  /**
   * Set the value of a scalar variable.  If a variable with this name
   * exists, then its value will be set to the new value.  If there is not
   * already a variable with this name, variableName will be added to the
   * list of variables, and its value will be set to the new value.
   */
  void SetScalarVariableValue(const char* variableName, double value);
  void SetScalarVariableValue(int i, double value);
  //@}

  //@{
  /**
   * Get the value of a scalar variable.
   */
  double GetScalarVariableValue(const char* variableName);
  double GetScalarVariableValue(int i);
  //@}

  //@{
  /**
   * Set the value of a vector variable.  If a variable with this name
   * exists, then its value will be set to the new value.  If there is not
   * already a variable with this name, variableName will be added to the
   * list of variables, and its value will be set to the new value.
   */
  void SetVectorVariableValue(
    const char* variableName, double xValue, double yValue, double zValue);
  void SetVectorVariableValue(const char* variableName, const double values[3])
  {
    this->SetVectorVariableValue(variableName, values[0], values[1], values[2]);
  }
  void SetVectorVariableValue(int i, double xValue, double yValue, double zValue);
  void SetVectorVariableValue(int i, const double values[3])
  {
    this->SetVectorVariableValue(i, values[0], values[1], values[2]);
  }
  //@}

  //@{
  /**
   * Get the value of a vector variable.
   */
  double* GetVectorVariableValue(const char* variableName) SVTK_SIZEHINT(3);
  void GetVectorVariableValue(const char* variableName, double value[3])
  {
    double* r = this->GetVectorVariableValue(variableName);
    value[0] = r[0];
    value[1] = r[1];
    value[2] = r[2];
  }
  double* GetVectorVariableValue(int i) SVTK_SIZEHINT(3);
  void GetVectorVariableValue(int i, double value[3])
  {
    double* r = this->GetVectorVariableValue(i);
    value[0] = r[0];
    value[1] = r[1];
    value[2] = r[2];
  }
  //@}

  /**
   * Get the number of scalar variables.
   */
  int GetNumberOfScalarVariables() { return static_cast<int>(this->ScalarVariableNames.size()); }

  /**
   * Get scalar variable index or -1 if not found
   */
  int GetScalarVariableIndex(const char* name);

  /**
   * Get the number of vector variables.
   */
  int GetNumberOfVectorVariables() { return static_cast<int>(this->VectorVariableNames.size()); }

  /**
   * Get scalar variable index or -1 if not found
   */
  int GetVectorVariableIndex(const char* name);

  /**
   * Get the ith scalar variable name.
   */
  const char* GetScalarVariableName(int i);

  /**
   * Get the ith vector variable name.
   */
  const char* GetVectorVariableName(int i);

  //@{
  /**
   * Returns whether a scalar variable is needed for the function evaluation.
   * This is only valid after a successful Parse(). Thus, call GetScalarResult()
   * or IsScalarResult() or similar method before calling this.
   */
  bool GetScalarVariableNeeded(int i);
  bool GetScalarVariableNeeded(const char* variableName);
  //@}

  //@{
  /**
   * Returns whether a vector variable is needed for the function evaluation.
   * This is only valid after a successful Parse(). Thus, call GetVectorResult()
   * or IsVectorResult() or similar method before calling this.
   */
  bool GetVectorVariableNeeded(int i);
  bool GetVectorVariableNeeded(const char* variableName);
  //@}

  /**
   * Remove all the current variables.
   */
  void RemoveAllVariables();

  /**
   * Remove all the scalar variables.
   */
  void RemoveScalarVariables();

  /**
   * Remove all the vector variables.
   */
  void RemoveVectorVariables();

  //@{
  /**
   * When ReplaceInvalidValues is on, all invalid values (such as
   * sqrt(-2), note that function parser does not handle complex
   * numbers) will be replaced by ReplacementValue. Otherwise an
   * error will be reported
   */
  svtkSetMacro(ReplaceInvalidValues, svtkTypeBool);
  svtkGetMacro(ReplaceInvalidValues, svtkTypeBool);
  svtkBooleanMacro(ReplaceInvalidValues, svtkTypeBool);
  svtkSetMacro(ReplacementValue, double);
  svtkGetMacro(ReplacementValue, double);
  //@}

  /**
   * Check the validity of the function expression.
   */
  void CheckExpression(int& pos, char** error);

  /**
   * Allow the user to force the function to be re-parsed
   */
  void InvalidateFunction();

protected:
  svtkFunctionParser();
  ~svtkFunctionParser() override;

  int Parse();

  /**
   * Evaluate the function, returning true on success, false on failure.
   */
  bool Evaluate();

  int CheckSyntax();

  void CopyParseError(int& position, char** error);

  void RemoveSpaces();
  char* RemoveSpacesFrom(const char* variableName);
  int OperatorWithinVariable(int idx);

  int BuildInternalFunctionStructure();
  void BuildInternalSubstringStructure(int beginIndex, int endIndex);
  void AddInternalByte(unsigned int newByte);

  int IsSubstringCompletelyEnclosed(int beginIndex, int endIndex);
  int FindEndOfMathFunction(int beginIndex);
  int FindEndOfMathConstant(int beginIndex);

  int IsVariableName(int currentIndex);
  int IsElementaryOperator(int op);

  int GetMathFunctionNumber(int currentIndex);
  int GetMathFunctionNumberByCheckingParenthesis(int currentIndex);
  int GetMathFunctionStringLength(int mathFunctionNumber);
  int GetMathConstantNumber(int currentIndex);
  int GetMathConstantStringLength(int mathConstantNumber);
  unsigned char GetElementaryOperatorNumber(char op);
  unsigned int GetOperandNumber(int currentIndex);
  int GetVariableNameLength(int variableNumber);

  int DisambiguateOperators();

  /**
   * Collects meta-data about which variables are needed by the current
   * function. This is called only after a successful call to this->Parse().
   */
  void UpdateNeededVariables();

  svtkSetStringMacro(ParseError);

  int FindPositionInOriginalFunction(const int& pos);

  char* Function;
  char* FunctionWithSpaces;

  int FunctionLength;
  std::vector<std::string> ScalarVariableNames;
  std::vector<std::string> VectorVariableNames;
  std::vector<double> ScalarVariableValues;
  std::vector<svtkTuple<double, 3> > VectorVariableValues;
  std::vector<bool> ScalarVariableNeeded;
  std::vector<bool> VectorVariableNeeded;

  std::vector<unsigned int> ByteCode;
  int ByteCodeSize;
  double* Immediates;
  int ImmediatesSize;
  double* Stack;
  int StackSize;
  int StackPointer;

  svtkTimeStamp FunctionMTime;
  svtkTimeStamp ParseMTime;
  svtkTimeStamp VariableMTime;
  svtkTimeStamp EvaluateMTime;
  svtkTimeStamp CheckMTime;

  svtkTypeBool ReplaceInvalidValues;
  double ReplacementValue;

  int ParseErrorPositon;
  char* ParseError;

private:
  svtkFunctionParser(const svtkFunctionParser&) = delete;
  void operator=(const svtkFunctionParser&) = delete;
};

#endif
