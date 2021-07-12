#ifndef METAWARENN_EXE_UTILS_H_
#define METAWARENN_EXE_UTILS_H_

#include <assert.h>
#define BYTE_ALIGNMENT 4

namespace metawarenn {

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline constexpr T isPowerOfTwo(T val) {
    return (val > 0) && ((val & (val - 1)) == 0);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline T alignVal(T val, T align) {
    assert(isPowerOfTwo(align));
    return (val + (align - 1)) & ~(align - 1);
}

} //namespace metawarenn
#endif //METAWARENN_EXE_UTILS_H_
