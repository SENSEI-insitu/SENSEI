#include <svtkMutableDirectedGraph.h>
#include <svtkNew.h>
#include <svtkTree.h>

#include <vector>

#include "svtkTreeDFSIterator.h"

int TestTreeDFSIterator(int, char*[])
{
  svtkNew<svtkMutableDirectedGraph> g;

  // Create vertices:
  svtkIdType v0 = g->AddVertex(); // Level 0
  svtkIdType v1 = g->AddVertex(); // Level 1
  svtkIdType v2 = g->AddVertex(); // Level 2
  svtkIdType v3 = g->AddVertex(); // Level 2
  svtkIdType v4 = g->AddVertex(); // Level 1
  svtkIdType v5 = g->AddVertex(); // Level 2
  svtkIdType v6 = g->AddVertex(); // Level 1
  svtkIdType v7 = g->AddVertex(); // Level 2
  svtkIdType v8 = g->AddVertex(); // Level 3

  // create a fully connected graph
  g->AddEdge(v0, v1);
  g->AddEdge(v1, v2);
  g->AddEdge(v1, v3);
  g->AddEdge(v0, v4);
  g->AddEdge(v4, v5);
  g->AddEdge(v0, v6);
  g->AddEdge(v6, v7);
  g->AddEdge(v7, v8);

  svtkNew<svtkTree> tree;
  tree->CheckedShallowCopy(g);

  std::vector<int> correctSequence;
  for (int i = 0; i <= 8; i++)
  {
    correctSequence.push_back(i);
  }

  svtkNew<svtkTreeDFSIterator> dfsIterator;
  dfsIterator->SetTree(tree);

  if (dfsIterator->GetStartVertex() != tree->GetRoot())
  {
    cout << "StartVertex is not defaulting to root" << endl;
    return EXIT_FAILURE;
  }

  // traverse the tree in a depth first fashion
  for (size_t i = 0; i < correctSequence.size(); i++)
  {
    if (!dfsIterator->HasNext())
    {
      cout << "HasNext() returned false before the end of the tree" << endl;
      return EXIT_FAILURE;
    }

    svtkIdType nextVertex = dfsIterator->Next();
    if (nextVertex != correctSequence[i])
    {
      cout << "Next vertex should be " << correctSequence[i] << " but it is " << nextVertex << endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
