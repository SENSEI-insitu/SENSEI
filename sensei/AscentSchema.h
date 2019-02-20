#ifndef ASCENT_SCHEMA_H
#define ASCENT_SCHEMA_H

#include <conduit.hpp>
#include <pugixml.hpp>

namespace sensei
{
  int AscentGetScene(pugi::xml_node &node, conduit::Node &actions);
  int AscentGetPipeline(pugi::xml_node &node, conduit::Node &actions);
};

#endif
