#!/usr/bin/env python2
import os, sys
import re

def displayHelp():
    print """
Usage: WhatModulesSVTK.py svtkSourceTree applicationFile|applicationFolder
  Generate a FindPackage(SVTK COMPONENTS) that lists all modules
  referenced by a set of files.

    For example:
      Running from the SVTK source,
        ./Utilities/Maintenance/WhatModulesSVTK.py . Filters/Modeling/Testing/Cxx/TestRotationalExtrusion.cxx
      Produces
        All modules referenced in the files:
        find_package(SVTK COMPONENTS
          CommonCore
          FiltersCore
          FiltersModeling
          FiltersSources
          RenderingCore
          RenderingOpenGL2
          TestingCore
          TestingRendering
        )
        Your application code includes 8 of 170 svtk modules.

"""
    exit(0)

def IncludesToPaths(path):
    '''
    Build a dict that maps include files to paths.
    '''
    includeToPath = dict()
    prog = re.compile(r"((?:svtk|QSVTK).*\.h)")
    for root, dirs, files in os.walk(path):
        for f in files:
            if prog.match(f):
                includeFile = prog.findall(f)[0]
                includeToPath[includeFile] = root
    return includeToPath

def FindModules(path):
    '''
    Build a dict that maps paths to modules.
    '''
    pathToModule = dict()
    fileProg = re.compile(r"svtk\.module$")
    for root, dirs, files in os.walk(path):
        for f in files:
            if fileProg.match(f):
                with open(os.path.join(root, f), "r") as fid:
                    contents = fid.read()
                args = contents.split()
                try:
                    idx = args.index('NAME')
                except ValueError:
                    raise RuntimeError('%s is missing a NAME field' % os.path.join(root, f))
                pathToModule[root] = args[idx + 1]
    return pathToModule

def FindIncludes(path):
    '''
    Build a set that contains svtk includes.
    '''
    includes = set()
    includeProg = re.compile(r"((?:svtk|QSVTK).*\.h)")
    with open(path, "r") as fid:
        contents = fid.read()
    incs = includeProg.findall(contents)
    includes.update(incs)
    return includes

def FindModuleFiles(path):
    '''
    Get a list of module files in the SVTK directory.
    '''
    moduleFiles = [os.path.join(root, name)
                 for root, dirs, files in os.walk(path)
                 for name in files
                 if name == ("svtk.module")]
    return moduleFiles

def MakeFindPackage(modules):
    '''
    Make a useful find_package command.
    '''
    # Print a useful cmake command
    res = "find_package(SVTK COMPONENTS\n"
    for module in sorted(modules):
        res +=  "  " + module.replace('SVTK::', '') + "\n"
    res +=  ")"
    return res

from pprint import pprint as pp

def main(svtkSourceDir, sourceFiles):
    '''
    Start the program
    '''
    # Generate dict's for mapping includes to modules
    includesToPaths = IncludesToPaths(svtkSourceDir + "/")
    pathsToModules = FindModules(svtkSourceDir + "/")

    # Test to see if SVTK source is provided
    if len(pathsToModules) == 0:
        raise IOError, svtkSourceDir +\
        " is not a SVTK source directory. It does not contain any svtk.module files."

    # Parse the module files making a dictionary of each module and its
    # dependencies or what it implements.
    moduleFiles = FindModuleFiles(svtkSourceDir + "/")

    # Build a set of includes for all command line files
    allIncludes = set()
    for f in sourceFiles:
        if os.path.isfile(f):
            allIncludes.update(FindIncludes(f))
        else:
            # We have a folder so look through all the files.
            for path, dirs, files in os.walk(f):
                for fn in files:
                    allIncludes.update(FindIncludes(os.path.join(path,fn)))
    if len(allIncludes) == 0:
        raise IOError, f + " does not exist"

    # Build a set that contains all modules referenced in command line files
    allModules = set()
    for inc in allIncludes:
        if inc in includesToPaths:
            module = includesToPaths[inc]
            if module in pathsToModules:
                allModules.add(pathsToModules[includesToPaths[inc]])

    modules = {'All modules referenced in the files:': allModules,
              }
    for k, v in modules.iteritems():
        print k
        print MakeFindPackage(v)
        print "Your application code includes " + str(len(v)) +\
              " of " + str(len(pathsToModules)) + " svtk modules."

if __name__ == '__main__':
    if len(sys.argv) < 3:
      displayHelp()
      exit(0)
    main(sys.argv[1], sys.argv[2:])
    print('This program is deprecated.')
    print('Please consider using FindNeededModules.py instead.')
