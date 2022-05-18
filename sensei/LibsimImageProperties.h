#ifndef sensei_LibsimImageProperties_h
#define sensei_LibsimImageProperties_h

#include "senseiConfig.h"

#include <string>

namespace sensei
{

class SENSEI_EXPORT LibsimImageProperties
{
public:
    LibsimImageProperties();
    ~LibsimImageProperties();

    void SetFilename(const std::string &s);
    void SetWidth(int val);
    void SetHeight(int val);
    void SetFormat(const std::string &s);

    const std::string &GetFilename() const;
    const std::string &GetFormat() const;
    int GetWidth() const;
    int GetHeight() const;

private:
    std::string filename;
    std::string format;
    int width, height;
};

}

#endif
