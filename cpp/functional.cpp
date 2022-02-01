#include <cstdio>
#include <iostream>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include "xoodoo.hpp"

#define RANDOM_UINT32(x, s_x, s_y, s_z) \
    uint32_t t;                         \
    s_x ^= s_x << 16;                   \
    s_x ^= s_x >> 5;                    \
    s_x ^= s_x << 1;                    \
    t = s_x;                            \
    s_x = s_y;                          \
    s_y = s_z;                          \
    s_z = t ^ s_x ^ s_y;                \
    x = s_z;

#define RANDOM_XOODOO_STATE(x, sx, sy, sz)      \
    for(int i = 0; i < NROWS; ++i)              \
    {                                           \
        for(int j = 0; j < NCOLS; ++j)          \
        {                                       \
            RANDOM_UINT32(x[i][j], sx, sy, sz); \
        }                                       \
    }

#define RANDOM_PAIR(x, xp, input_diff, sx, sy, sz)  \
    RANDOM_XOODOO_STATE(x, sx, sy, sz)              \
    for(int i = 0; i < NROWS; ++i)                  \
    {                                               \
        for(int j = 0; j < NCOLS; ++j)              \
        {                                           \
            xp[i][j] = x[i][j] ^ input_diff[i][j];  \
        }                                           \
    }

inline void zeroize_bit(uint32_t & x, int i)
{
    x &= ((0xFFFFFFFF) ^ (uint32_t{1} << i));
}

inline bool ith_bit(uint32_t x, int i)
{
    return (x & (1 << i)) >> i;
}

inline bool is_functional_pair(const uint32_t x[NROWS][NCOLS], const uint32_t y[NROWS][NCOLS])
{
    uint32_t tx, ty;

    //y[0][0]
    {
        tx = x[0][0]; ty = y[0][0];
        zeroize_bit(tx, 5); zeroize_bit(tx, 14);
        zeroize_bit(ty, 5); zeroize_bit(ty, 14);

        bool y000005 = ith_bit(y[0][0], 5);
        bool y000014 = ith_bit(y[0][0], 14);

        bool x000005 = ith_bit(x[0][0], 5);
        bool x010006 = ith_bit(x[1][0], 6);
        bool x020213 = ith_bit(x[2][2], 13);
        bool x000014 = ith_bit(x[0][0], 14);
        bool x010015 = ith_bit(x[1][0], 15);
        bool x020222 = ith_bit(x[2][2], 22);

        if (!(
                (tx == ty) &&
                (y000005 == ((x000005 & x010006) ^ x000005 ^ x010006 ^ x020213)) &&
                (y000014 == ((x000014 & x010015) ^ x000014 ^ x010015 ^ x020222))
            ))
            return false;
    }

    //y[0][1]
    {
        if (y[0][1] != (x[0][1] ^ 0x80000000))
            return false;
    }

    //y[0][2]
    {
        tx = x[0][2]; ty = y[0][2];
        zeroize_bit(tx, 15); zeroize_bit(tx, 24);
        zeroize_bit(ty, 15); zeroize_bit(ty, 24);

        bool y000215 = ith_bit(y[0][2], 15);
        bool y000224 = ith_bit(y[0][2], 24);

        bool x000215 = ith_bit(x[0][2], 15);
        bool x020023 = ith_bit(x[2][0], 23);
        bool x010216 = ith_bit(x[1][2], 16);
        bool x000224 = ith_bit(x[0][2], 24);
        bool x020000 = ith_bit(x[2][0], 0);
        bool x010225 = ith_bit(x[1][2], 25);

        if (!(
                (ty == (tx ^ 0x00002010)) &&
                (y000215 == ((x000215 & x020023) ^ x010216 ^ 1)) &&
                (y000224 == ((x000224 & x020000) ^ x010225 ^ 1))
            ))
            return false;
    }

    //y[0][3]
    {
        tx = x[0][3]; ty = y[0][3];
        zeroize_bit(tx, 0); zeroize_bit(tx, 4); zeroize_bit(tx, 13); zeroize_bit(tx, 16); zeroize_bit(tx, 25);
        zeroize_bit(ty, 0); zeroize_bit(ty, 4); zeroize_bit(ty, 13); zeroize_bit(ty, 16); zeroize_bit(ty, 25);

        bool y000300 = ith_bit(y[0][3], 0);
        bool y000304 = ith_bit(y[0][3], 4);
        bool y000313 = ith_bit(y[0][3], 13);
        bool y000316 = ith_bit(y[0][3], 16);
        bool y000325 = ith_bit(y[0][3], 25);

        bool x000300 = ith_bit(x[0][3], 0);
        bool x010301 = ith_bit(x[1][3], 1);
        bool x020108 = ith_bit(x[2][1], 8);
        bool x000304 = ith_bit(x[0][3], 4);
        bool x010305 = ith_bit(x[1][3], 5);
        bool x020112 = ith_bit(x[2][1], 12);
        bool x000313 = ith_bit(x[0][3], 13);
        bool x010314 = ith_bit(x[1][3], 14);
        bool x020121 = ith_bit(x[2][1], 21);
        bool x000316 = ith_bit(x[0][3], 16);
        bool x020124 = ith_bit(x[2][1], 24);
        bool x010317 = ith_bit(x[1][3], 17);
        bool x000325 = ith_bit(x[0][3], 25);
        bool x020101 = ith_bit(x[2][1], 1);
        bool x010326 = ith_bit(x[1][3], 26);

        if (!(
                (ty == (tx ^ 0x00004020) &&
                (y000300 == ((x000300 & x010301) ^ x000300 ^ x010301 ^ x020108)) &&
                (y000304 == ((x000304 & x010305) ^ x000304 ^ x010305 ^ x020112)) &&
                (y000313 == ((x000313 & x010314) ^ x000313 ^ x010314 ^ x020121)) &&
                (y000316 == ((x000316 & x020124) ^ x010317 ^ 1)) &&
                (y000325 == ((x000325 & x020101) ^ x010326 ^ 1)))
            ))
            return false;
    }

    //y[1][0]
    {
        if (y[1][0] != (x[1][0] ^ 0x00008040))
            return false;
    }

    //y[1][1]
    {
        tx = x[1][1]; ty = y[1][1];
        zeroize_bit(tx, 0);
        zeroize_bit(ty, 0);

        bool y010100 = ith_bit(y[1][1], 0);

        bool x000131 = ith_bit(x[0][1], 31);
        bool x010100 = ith_bit(x[1][1], 0);
        bool x020307 = ith_bit(x[2][3], 7);

        if (!(
                (ty == tx) &&
                (y010100 == ((x000131 & x010100) ^ x020307 ^ 1))
            ))
            return false;
    }

    //y[1][2]
    {
        tx = x[1][2]; ty = y[1][2];
        zeroize_bit(tx, 5); zeroize_bit(tx, 14); zeroize_bit(tx, 16); zeroize_bit(tx, 25);
        zeroize_bit(ty, 5); zeroize_bit(ty, 14); zeroize_bit(ty, 16); zeroize_bit(ty, 25);

        bool y010205 = ith_bit(y[1][2], 5);
        bool y010214 = ith_bit(y[1][2], 14);
        bool y010216 = ith_bit(y[1][2], 16);
        bool y010225 = ith_bit(y[1][2], 25);

        bool x000204 = ith_bit(x[0][2], 4);
        bool x010205 = ith_bit(x[1][2], 5);
        bool x020012 = ith_bit(x[2][0], 12);
        bool x000213 = ith_bit(x[0][2], 13);
        bool x010214 = ith_bit(x[1][2], 14);
        bool x020021 = ith_bit(x[2][0], 21);
        bool x000215 = ith_bit(x[0][2], 15);
        bool x010216 = ith_bit(x[1][2], 16);
        bool x020023 = ith_bit(x[2][0], 23);
        bool x000224 = ith_bit(x[0][2], 24);
        bool x010225 = ith_bit(x[1][2], 25);
        bool x020000 = ith_bit(x[2][0], 0);

        if (!(
                (ty == tx) &&
                (y010205 == ((x000204 & x010205) ^ x020012 ^ 1)) &&
                (y010214 == ((x000213 & x010214) ^ x020021 ^ 1)) &&
                (y010216 == ((x010216 & x020023) ^ x000215 ^ x010216 ^ x020023)) &&
                (y010225 == ((x010225 & x020000) ^ x000224 ^ x010225 ^ x020000))
            ))
            return false;
    }

    //y[1][3]
    {
        tx = x[1][3]; ty = y[1][3];
        zeroize_bit(tx, 6); zeroize_bit(tx, 15); zeroize_bit(tx, 17); zeroize_bit(tx, 26);
        zeroize_bit(ty, 6); zeroize_bit(ty, 15); zeroize_bit(ty, 17); zeroize_bit(ty, 26);

        bool y010306 = ith_bit(y[1][3], 6);
        bool y010315 = ith_bit(y[1][3], 15);
        bool y010317 = ith_bit(y[1][3], 17);
        bool y010326 = ith_bit(y[1][3], 26);

        bool x000305 = ith_bit(x[0][3], 5);
        bool x010306 = ith_bit(x[1][3], 6);
        bool x020113 = ith_bit(x[2][1], 13);
        bool x000314 = ith_bit(x[0][3], 14);
        bool x010315 = ith_bit(x[1][3], 15);
        bool x020122 = ith_bit(x[2][1], 22);
        bool x000316 = ith_bit(x[0][3], 16);
        bool x010317 = ith_bit(x[1][3], 17);
        bool x020124 = ith_bit(x[2][1], 24);
        bool x000325 = ith_bit(x[0][3], 25);
        bool x010326 = ith_bit(x[1][3], 26);
        bool x020101 = ith_bit(x[2][1], 1);

        if (!(
                (ty == (tx ^ 0x00004022)) &&
                (y010306 == ((x000305 & x010306) ^ x020113 ^ 1)) &&
                (y010315 == ((x000314 & x010315) ^ x020122 ^ 1)) &&
                (y010317 == ((x010317 & x020124) ^ x000316 ^ x010317 ^ x020124)) &&
                (y010326 == ((x010326 & x020101) ^ x000325 ^ x010326 ^ x020101))
            ))
            return false;
    }

    //y[2][0]
    {
        tx = x[2][0]; ty = y[2][0];
        zeroize_bit(tx, 12); zeroize_bit(tx, 21);
        zeroize_bit(ty, 12); zeroize_bit(ty, 21);

        bool y020012 = ith_bit(y[2][0], 12);
        bool y020021 = ith_bit(y[2][0], 21);

        bool x000204 = ith_bit(x[0][2], 4);
        bool x020012 = ith_bit(x[2][0], 12);
        bool x010205 = ith_bit(x[1][2], 5);
        bool x000213 = ith_bit(x[0][2], 13);
        bool x020021 = ith_bit(x[2][0], 21);
        bool x010214 = ith_bit(x[1][2], 14);

        if (!(
                (ty == (tx ^ 0x00800001)) &&
                (y020012 == ((x000204 & x020012) ^ x000204 ^ x010205 ^ x020012)) &&
                (y020021 == ((x000213 & x020021) ^ x000213 ^ x010214 ^ x020021))
            ))
            return false;
    }

    //y[2][1]
    {
        tx = x[2][1]; ty = y[2][1];
        zeroize_bit(tx, 8); zeroize_bit(tx, 12); zeroize_bit(tx, 13); zeroize_bit(tx, 21); zeroize_bit(tx, 22);
        zeroize_bit(ty, 8); zeroize_bit(ty, 12); zeroize_bit(ty, 13); zeroize_bit(ty, 21); zeroize_bit(ty, 22);

        bool y020108 = ith_bit(y[2][1], 8);
        bool y020112 = ith_bit(y[2][1], 12);
        bool y020113 = ith_bit(y[2][1], 13);
        bool y020121 = ith_bit(y[2][1], 21);
        bool y020122 = ith_bit(y[2][1], 22);

        bool x000300 = ith_bit(x[0][3], 0);
        bool x010301 = ith_bit(x[1][3], 1);
        bool x020108 = ith_bit(x[2][1], 8);
        bool x000304 = ith_bit(x[0][3], 4);
        bool x010305 = ith_bit(x[1][3], 5);
        bool x020112 = ith_bit(x[2][1], 12);
        bool x000305 = ith_bit(x[0][3], 5);
        bool x020113 = ith_bit(x[2][1], 13);
        bool x010306 = ith_bit(x[1][3], 6);
        bool x000313 = ith_bit(x[0][3], 13);
        bool x010314 = ith_bit(x[1][3], 14);
        bool x020121 = ith_bit(x[2][1], 21);
        bool x000314 = ith_bit(x[0][3], 14);
        bool x020122 = ith_bit(x[2][1], 22);
        bool x010315 = ith_bit(x[1][3], 15);

        if (!(
                (ty == (tx ^ 0x01000002)) &&
                (y020108 == ((x010301 & x020108) ^ x000300 ^ 1)) &&
                (y020112 == ((x010305 & x020112) ^ x000304 ^  1)) &&
                (y020113 == ((x000305 & x020113) ^ x000305 ^ x010306 ^ x020113)) &&
                (y020121 == ((x010314 & x020121) ^ x000313 ^ 1)) &&
                (y020122 == ((x000314 & x020122) ^ x000314 ^ x010315 ^ x020122))
            ))
            return false;
    }

    //y[2][2]
    {
        tx = x[2][2]; ty = y[2][2];
        zeroize_bit(tx, 13); zeroize_bit(tx, 22);
        zeroize_bit(ty, 13); zeroize_bit(ty, 22);

        bool y020213 = ith_bit(y[2][2], 13);
        bool y020222 = ith_bit(y[2][2], 22);

        bool x000005 = ith_bit(x[0][0], 5);
        bool x010006 = ith_bit(x[1][0], 6);
        bool x020213 = ith_bit(x[2][2], 13);
        bool x000014 = ith_bit(x[0][0], 14);
        bool x010015 = ith_bit(x[1][0], 15);
        bool x020222 = ith_bit(x[2][2], 22);

        if (!(
                (ty == tx) &&
                (y020213 == ((x010006 & x020213) ^ x000005 ^ 1)) &&
                (y020222 == ((x010015 & x020222) ^ x000014 ^ 1))
            ))
            return false;
    }

    //y[2][3]
    {
        tx = x[2][3]; ty = y[2][3];
        zeroize_bit(tx, 7);
        zeroize_bit(ty, 7);

        bool y020307 = ith_bit(y[2][3], 7);

        bool x000131 = ith_bit(x[0][1], 31);
        bool x020307 = ith_bit(x[2][3], 7);
        bool x010100 = ith_bit(x[1][1], 0);

        if (!(
                (ty == tx) &&
                (y020307 == ((x000131 & x020307) ^ x000131 ^ x010100 ^ x020307))
            ))
            return false;
    }

    return true;
}


int main() {
    uint64_t nthreads = std::thread::hardware_concurrency();

    const uint64_t ntrials = 100;
    const uint64_t begin_exponent = 1;
    const uint64_t end_exponent = 45;

    for(uint64_t exponent = begin_exponent; exponent <= end_exponent; ++exponent)
    {
        uint64_t nsamples = uint64_t{1} << exponent;
        uint64_t nsuccess = 0;

        for(uint64_t _ = 0; _ < ntrials; ++_)
        {
            unsigned long long counter = 0;

#pragma omp parallel reduction(+:counter) num_threads(nthreads) firstprivate(nthreads, nsamples) default(none)
            {
                const uint32_t input_diff[NROWS][NCOLS] = {
                        {0x1D28B03E, 0x09199081, 0x46125265, 0x56D31D2C},
                        {0x5D28B03E, 0x89199081, 0x46125265, 0x56D31D2C},
                        {0x1D28B03E, 0x89119081, 0x46125265, 0x56D31D2C}
                };

                int fd = open("/dev/urandom", O_RDONLY);
                if (fd < 0) {
                    printf("unable to open /dev/urandom\n");
                    exit(1);
                }

                uint32_t state_x, state_y, state_z;
                read(fd, &state_x, sizeof(state_x));
                read(fd, &state_y, sizeof(state_y));
                read(fd, &state_z, sizeof(state_z));
                close(fd);

                uint32_t x[NROWS][NCOLS] = {}, xp[NROWS][NCOLS] = {};

                for (uint64_t _ = 0; _ < nsamples / nthreads; _++)
                {
                    RANDOM_PAIR(x, xp, input_diff, state_x, state_y, state_z)

                    xoodoo<3>(x);
                    xoodoo<3>(xp);

                    if (is_functional_pair(x, xp))
                        ++counter;
                }
            }
            if (counter > 0)
                ++nsuccess;
        }

        std::cout << exponent << " " << nsuccess << std::endl;
    }

    return 0;
}
