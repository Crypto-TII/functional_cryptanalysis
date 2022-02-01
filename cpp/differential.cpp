/*
Copyright 2022 Technology Innovation Institute LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstring>
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

#define OUTPUT_DIFF(D, x, xp)               \
    for(int i = 0; i < NROWS; ++i)          \
    {                                       \
        for(int j = 0; j < NCOLS; ++j)      \
        {                                   \
            D[i][j] = x[i][j] ^ xp[i][j];   \
        }                                   \
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

                const uint32_t output_diff[NROWS][NCOLS] = {
                        {0x00000000, 0x80000000, 0x01002010, 0x02006031},
                        {0x00008040, 0x00000000, 0x00014020, 0x00024022},
                        {0x00800001, 0x01402002, 0x00402000, 0x00000080}
                };

                int fd = open("/dev/urandom", O_RDONLY);
                if (fd < 0)
                {
                    printf("unable to open /dev/urandom\n");
                    exit(1);
                }

                uint32_t state_x, state_y, state_z;
                read(fd, &state_x, sizeof(state_x));
                read(fd, &state_y, sizeof(state_y));
                read(fd, &state_z, sizeof(state_z));
                close(fd);

                uint32_t x[NROWS][NCOLS] = {}, xp[NROWS][NCOLS] = {};
                uint32_t D[NROWS][NCOLS] = {};

                for(uint64_t _ = 0; _ < nsamples / nthreads; _++)
                {
                    RANDOM_PAIR(x, xp, input_diff, state_x, state_y, state_z)

                    xoodoo<3>(x);
                    xoodoo<3>(xp);

                    OUTPUT_DIFF(D, x, xp)

                    if (memcmp(D, output_diff, sizeof(uint32_t)*NROWS*NCOLS) == 0)
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
