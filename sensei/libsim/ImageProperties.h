#ifndef sensei_libsim_ImageProperties_h
#define sensei_libsim_ImageProperties_h
#include <string>

namespace sensei
{
namespace libsim
{

class ImageProperties
{
public:
    ImageProperties();
    ~ImageProperties();
    
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
    int         width, height;
};

} // libsim
} // sensei
#endif
