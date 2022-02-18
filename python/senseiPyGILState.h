#ifndef senseiPyGILState_h
#define senseiPyGILState_h

#include <Python.h>

// RAII helper for managing the Python GIL
// The class aquires the GIL during contruction
// and releases during destruction.
class SENSEI_EXPORT senseiPyGILState
{
public:
    senseiPyGILState()
    { m_state = PyGILState_Ensure(); }

    ~senseiPyGILState()
    { PyGILState_Release(m_state); }

    senseiPyGILState(const senseiPyGILState&) = delete;
    void operator=(const senseiPyGILState&) = delete;

private:
    PyGILState_STATE m_state;
};

#endif
