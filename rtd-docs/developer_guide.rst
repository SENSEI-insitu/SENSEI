Developer Guidelines
====================
Git workflow
-------------



Code style
----------
Here are some of the guidelines

* use 2 spaces for indentation, no tabs or trailing white space
* use `CamelCase` for names
* variable names should be descriptive, with in reason
* class member variables, methods, and namespace functions start with an upper case character, free functions start with a lower case
* use the this pointer to access class member variables and methods
* generally operators should be separated from operands by a single white space
* for loop and conditional braces are indented 2 spaces, and the contained code is written at the same indentation level as the braces
* functions and class braces are not indented, but contained code is indented 2 spaces.
* a comment containing one space and 77 - precede class method definitions
* pointers and reference markers should be preceded by a space.
* generally wrap code at 80 chars
* treat warnings as errors, compile with -Wall and clean all warnings
* avoid 1 line conditionals

there are surely other details to this style, but I think that is the crux of it.

and a snippet to illustrate:


.. code-block:: C++

    // a free function
    void fooBar()
    {
      printf("blah");
    }

    // a namespace
    namespace myNs
    {
    // with a function
    void Foo()
    {
      pritf("Foo");
    }

    }

    // a class
    class Foo
    {
    public:
      // CamelCase methods and members
      void SetBar(int bar);
      int GetBar();

      // pointer and reference arguments
      void GetBar(int &bar);
      void GetBar(int *bar);

    private:
      int Bar;
    };

    // ---------------------------------------------------------------------------
    void Foo::SetBar(int bar)
    {
      // a member function
      this->Bar = bar;
    }

    // ---------------------------------------------------------------------------
    int Foo::GetBar()
    {
      return this->Bar;
    }

    int main(int argc, char **argv)
    {
      // a conditional
      if (strcmp("foo", argv[1]) == 0)
        {
        foo();
        }
      else
        {
        bar();
        }
      return 0;
    }

Regressions tests
-----------------

