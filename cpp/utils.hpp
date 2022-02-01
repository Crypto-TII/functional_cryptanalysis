#ifndef FUNCTIONAL_UTILS_HPP
#define FUNCTIONAL_UTILS_HPP

inline uint32_t rotate_left32(uint32_t x, int nbits)
{
    return (x << nbits) | (x >> (32 - nbits));
}

#endif
