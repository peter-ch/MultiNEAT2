#ifndef MULTINEAT_FILE_IO_H
#define MULTINEAT_FILE_IO_H

#include <cstdio>

namespace NEAT
{
namespace detail
{

inline std::FILE* OpenFile(const char* filename, const char* mode)
{
    if (filename == nullptr || mode == nullptr)
    {
        return nullptr;
    }

#ifdef _MSC_VER
    std::FILE* file = nullptr;
    return fopen_s(&file, filename, mode) == 0 ? file : nullptr;
#else
    return std::fopen(filename, mode);
#endif
}

} // namespace detail
} // namespace NEAT

#endif // MULTINEAT_FILE_IO_H
