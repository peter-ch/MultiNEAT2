#ifndef MULTINEAT_LEGACY_ASSERT_H
#define MULTINEAT_LEGACY_ASSERT_H

// This filename is part of MultiNEAT's historical public interface. On
// case-insensitive filesystems it also shadows the C runtime's <assert.h>.
// Define the standard macro here so <cassert>, Python, and pybind11 continue
// to work when the MultiNEAT include directory appears first on the path.
#include <cstdlib>
#include <iostream>

#ifndef assert
#  ifdef NDEBUG
#    define assert(expression) ((void)0)
#  else
#    define assert(expression)                                                \
        ((expression)                                                        \
             ? static_cast<void>(0)                                          \
             : (std::cerr << "Assertion failed: " #expression << " ("       \
                          << __FILE__ << ':' << __LINE__ << ")\n",            \
                std::abort()))
#  endif
#endif

#ifdef ASSERT
#  undef ASSERT
#endif

#ifdef VERIFY
#  undef VERIFY
#endif

// Preserve the established release behavior of ASSERT while ensuring VERIFY
// always evaluates its expression. This avoids silently dropping side effects.
#ifdef NDEBUG
#  define ASSERT(expression) ((void)0)
#  define VERIFY(expression) static_cast<void>(expression)
#else
#  define ASSERT(expression) assert(expression)
#  define VERIFY(expression) assert(expression)
#endif

#endif
