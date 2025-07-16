#ifndef ASSERT_H
#define ASSERT_H

#include <iostream>
#include <cassert>
#include <exception>

// kill any existing declarations
#ifdef ASSERT
#undef ASSERT
#endif

#ifdef VERIFY
#undef VERIFY
#endif

#ifdef DEBUG

#if 1

//--------------
//  debug macros
//--------------
#define BREAK_CPU()            //__asm { int 3 }

#define ASSERT(expr)\
        {\
            if( !(expr) )\
            {\
                std::cout << "\n*** ASSERT! ***\n" << \
                __FILE__ ", line " << __LINE__ << ": " << \
                #expr << " is false\n\n";\
                throw std::exception();\
            }\
        }

#define VERIFY(expr)\
        {\
            if( !(expr) )\
            {\
                std::cout << "\n*** VERIFY FAILED ***\n" << \
                __FILE__ ", line " << __LINE__ << ": " << \
                #expr << " is false\n\n";\
                BREAK_CPU();\
            }\
        }
#else

#define ASSERT(expr)\
        {\
            if( !(expr) )\
            {\
                std::cout << "\n*** ASSERT ***\n"; \
                assert(expr);\
            }\
        }


#define VERIFY(expr)\
        {\
            if( !(expr) )\
            {\
                std::cout << "\n*** VERIFY FAILED ***\n"; \
                assert(expr);\
            }\
        }

#endif

#else // _DEBUG

//--------------
//  release macros
//--------------

// ASSERT gets optimised out completely
#define ASSERT(expr)

// verify has expression evaluated, but no further action taken
#define VERIFY(expr) //if( expr ) {}

#endif

#endif // INCLUDE_GUARD_Assert_h
