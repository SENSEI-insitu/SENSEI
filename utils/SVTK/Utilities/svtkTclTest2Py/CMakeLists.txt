IF(PYTHON_EXECUTABLE AND SVTK_WRAP_PYTHON)
  SET(SVTK_PYTHON_FILES
    backdrop.py
    catch.py
    expr.py
    file.py
    info.py
    mccases.py
    rtImageTest.py
    )

  file(COPY ${SVTK_PYTHON_FILES} DESTINATION  ${SVTK_BINARY_DIR}/Utilities/svtkTclTest2Py/)
ENDIF()


