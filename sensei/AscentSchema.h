#ifndef ASCENT_SCHEMA_H
#define ASCENT_SCHEMA_H

#include "conduit.hpp"
#include "ascent.hpp"
#include <pugixml.hpp>

namespace sensei
{
  int GetScene(pugi::xml_node &node, conduit::Node &actions);
  int GetPipeline(pugi::xml_node &node, conduit::Node &actions);

  int GetClip(pugi::xml_node &node, conduit::Node &actions);
  int GetClipWithField(pugi::xml_node &node, conduit::Node &actions);
  int GetThreshold(pugi::xml_node &node, conduit::Node &actions);
  int GetContour(pugi::xml_node &node, conduit::Node &actions);
  int GetSlice(pugi::xml_node &node, conduit::Node &actions);
  int Get3Slice(pugi::xml_node &node, conduit::Node &actions);
};
#endif
