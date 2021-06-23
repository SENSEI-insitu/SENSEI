#include <svtkMutableDirectedGraph.h>
#include <svtkNew.h>
#include <svtkTree.h>

#include <vector>

#include "svtkTreeBFSIterator.h"

int TestTreeBFSIterator(int, char*[])
{
  svtkNew<svtkMutableDirectedGraph> g;

  // Create vertices:
  // Level 0
  svtkIdType v0 = g->AddVertex();
  // Level 1
  svtkIdType v1 = g->AddVertex();
  svtkIdType v2 = g->AddVertex();
  // Level 2
  svtkIdType v3 = g->AddVertex();
  svtkIdType v4 = g->AddVertex();
  svtkIdType v5 = g->AddVertex();
  // Level 3
  svtkIdType v6 = g->AddVertex();
  svtkIdType v7 = g->AddVertex();
  svtkIdType v8 = g->AddVertex();

  // create a fully connected graph
  g->AddEdge(v0, v1);
  g->AddEdge(v0, v2);
  g->AddEdge(v1, v3);
  g->AddEdge(v2, v4);
  g->AddEdge(v2, v5);
  g->AddEdge(v4, v6);
  g->AddEdge(v4, v7);
  g->AddEdge(v5, v8);

  svtkNew<svtkTree> tree;
  tree->CheckedShallowCopy(g);

  std::vector<int> correctSequence;
  for (int i = 0; i <= 8; i++)
  {
    correctSequence.push_back(i);
  }

  svtkNew<svtkTreeBFSIterator> bfsIterator;
  bfsIterator->SetTree(tree);

  if (bfsIterator->GetStartVertex() != tree->GetRoot())
  {
    cout << "StartVertex is not defaulting to root" << endl;
    return EXIT_FAILURE;
  }

  // traverse the tree in a depth first fashion
  for (size_t i = 0; i < correctSequence.size(); i++)
  {
    if (!bfsIterator->HasNext())
    {
      cout << "HasNext() returned false before the end of the tree" << endl;
      return EXIT_FAILURE;
    }

    svtkIdType nextVertex = bfsIterator->Next();
    if (nextVertex != correctSequence[i])
    {
      cout << "Next vertex should be " << correctSequence[i] << " but it is " << nextVertex << endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
