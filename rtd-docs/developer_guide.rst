********************
Developer Guidelines
********************

Git workflow
============
When working on a feature for a future release make a pull request targeting
the develop branch. Only features to be included in the next release should be
merged.  A code review should be made prior to merge. New features should
always be accompanied with read the docs documentation and at least one
regression test.

We use the branching model described here:
https://nvie.com/posts/a-successful-git-branching-model/
and detailed below in case the above link goes away in the future.

The main branches
-----------------
The central repo holds two main branches with an infinite lifetime:

* master
* develop

The master branch at origin should be familiar to every Git user. Parallel to
the master branch, another branch exists called develop.

We consider origin/master to be the main branch where the source code of HEAD
always reflects a production-ready state.

We consider origin/develop to be the main branch where the source code of HEAD
always reflects a state with the latest delivered development changes for the
next release. Some would call this the "integration branch".

When the source code in the develop branch reaches a stable point and is ready
to be released, all of the changes should be merged back into master somehow
and then tagged with a release number. How this is done in detail will be
discussed further on.

Therefore, each time when changes are merged back into master, this is a new
production release by definition. We tend to be very strict at this, so that
theoretically, we could use a Git hook script to automatically build and
roll-out our software to our production servers every time there was a commit on
master.

Feature branches
----------------
May branch off from: develop
Must merge back into: develop
Branch naming convention: anything except master, develop, or release-*

Feature branches (or sometimes called topic branches) are used to develop new
features for the upcoming or a distant future release. When starting
development of a feature, the target release in which this feature will be
incorporated may well be unknown at that point. The essence of a feature branch
is that it exists as long as the feature is in development, but will eventually
be merged back into develop (to definitely add the new feature to the upcoming
release) or discarded (in case of a disappointing experiment).

Creating a feature branch
^^^^^^^^^^^^^^^^^^^^^^^^^

When starting work on a new feature, branch off from the develop branch.

.. code-block:: shell

    $ git checkout -b myfeature develop

Incorporating a finished feature on develop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finished features may be merged into the develop branch to definitely add them
to the upcoming release:

.. code-block:: shell

    $ git checkout develop
    $ git merge --no-ff myfeature
    $ git branch -d myfeature
    $ git push origin develop

The --no-ff flag causes the merge to always create a new commit object, even if
the merge could be performed with a fast-forward. This avoids losing
information about the historical existence of a feature branch and groups
together all commits that together added the feature.


Release branches
----------------
May branch off from: develop
Must merge back into: develop and master
Branch naming convention: release-*

Release branches support preparation of a new production release. They allow
for last-minute dotting of i’s and crossing t’s. Furthermore, they allow for
minor bug fixes and preparing meta-data for a release (version number,
build dates, etc.). By doing all of this work on a release branch, the
develop branch is cleared to receive features for the next big release.

The key moment to branch off a new release branch from develop is when develop
(almost) reflects the desired state of the new release. At least all features
that are targeted for the release-to-be-built must be merged in to develop at
this point in time. All features targeted at future releases may not—they must
wait until after the release branch is branched off.

It is exactly at the start of a release branch that the upcoming release gets
assigned a version number—not any earlier. Up until that moment, the develop
branch reflected changes for the "next release", but it is unclear whether that
“next release” will eventually become 0.3 or 1.0, until the release branch is
started. That decision is made on the start of the release branch and is
carried out by the project’s rules on version number bumping.

Creating a release branch
^^^^^^^^^^^^^^^^^^^^^^^^^
Release branches are created from the develop branch. For example, say version
1.1.5 is the current production release and we have a big release coming up.
The state of develop is ready for the “next release” and we have decided that
this will become version 1.2 (rather than 1.1.6 or 2.0). So we branch off and
give the release branch a name reflecting the new version number:

.. code-block:: shell

    $ git checkout -b release-1.2 develop
    $ ./bump-version.sh 1.2
    $ git commit -a -m "Bumped version number to 1.2"

After creating a new branch and switching to it, we bump the version number.
Here, bump-version.sh is a fictional shell script that changes some files in
the working copy to reflect the new version. (This can of course be a manual
change—the point being that some files change.) Then, the bumped version number
is committed.

This new branch may exist there for a while, until the release may be rolled
out definitely. During that time, bug fixes may be applied in this branch
(rather than on the develop branch). Adding large new features here is strictly
prohibited. They must be merged into develop, and therefore, wait for the next
big release.

Finishing a release branch
^^^^^^^^^^^^^^^^^^^^^^^^^^
When the state of the release branch is ready to become a real release, some
actions need to be carried out. First, the release branch is merged into master
(since every commit on master is a new release by definition, remember). Next,
that commit on master must be tagged for easy future reference to this
historical version. Finally, the changes made on the release branch need to be
merged back into develop, so that future releases also contain these bug fixes.

.. code-block:: shell

    $ git checkout master
    $ git merge --no-ff release-1.2
    $ git tag -a 1.2

The release is now done, and tagged for future reference.


Code style
==========
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
=================
New classes should be submitted with a regression test.


User guide code style
=====================
Please use this style https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html


