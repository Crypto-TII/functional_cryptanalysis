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

#ifndef FUNCTIONAL_XOODOO_HPP
#define FUNCTIONAL_XOODOO_HPP

#include "utils.hpp"

#define NROWS 3
#define NCOLS 4
#define WORD_SIZE 32

#define MAX_NROUNDS 12

const uint32_t C[] = {
        0x00000058,
        0x00000038,
        0x000003C0,
        0x000000D0,
        0x00000120,
        0x00000014,
        0x00000060,
        0x0000002C,
        0x00000380,
        0x000000F0,
        0x000001A0,
        0x00000012
};

template<std::size_t NROUNDS>
inline void xoodoo(uint32_t A[NROWS][NCOLS])
{
    uint32_t P[NCOLS];
    int r = 0;

    for (int r = MAX_NROUNDS - NROUNDS; r < MAX_NROUNDS; ++r)
    {
        P[0] = A[0][0] ^ A[1][0] ^ A[2][0];
        P[1] = A[0][1] ^ A[1][1] ^ A[2][1];
        P[2] = A[0][2] ^ A[1][2] ^ A[2][2];
        P[3] = A[0][3] ^ A[1][3] ^ A[2][3];

        A[0][0] ^= rotate_left32(P[3], 5) ^ rotate_left32(P[3], 14);
        A[0][1] ^= rotate_left32(P[0], 5) ^ rotate_left32(P[0], 14);
        A[0][2] ^= rotate_left32(P[1], 5) ^ rotate_left32(P[1], 14);
        A[0][3] ^= rotate_left32(P[2], 5) ^ rotate_left32(P[2], 14);

        A[1][0] ^= rotate_left32(P[3], 5) ^ rotate_left32(P[3], 14);
        A[1][1] ^= rotate_left32(P[0], 5) ^ rotate_left32(P[0], 14);
        A[1][2] ^= rotate_left32(P[1], 5) ^ rotate_left32(P[1], 14);
        A[1][3] ^= rotate_left32(P[2], 5) ^ rotate_left32(P[2], 14);

        A[2][0] ^= rotate_left32(P[3], 5) ^ rotate_left32(P[3], 14);
        A[2][1] ^= rotate_left32(P[0], 5) ^ rotate_left32(P[0], 14);
        A[2][2] ^= rotate_left32(P[1], 5) ^ rotate_left32(P[1], 14);
        A[2][3] ^= rotate_left32(P[2], 5) ^ rotate_left32(P[2], 14);

        /* rho west */
        P[0] = A[1][3];
        A[1][3] = A[1][2];
        A[1][2] = A[1][1];
        A[1][1] = A[1][0];
        A[1][0] = P[0];

        A[2][0] = rotate_left32(A[2][0], 11);
        A[2][1] = rotate_left32(A[2][1], 11);
        A[2][2] = rotate_left32(A[2][2], 11);
        A[2][3] = rotate_left32(A[2][3], 11);

        /* iota */
        A[0][0] ^= C[r];

        /* chi */
        P[0] = ~A[1][0] & A[2][0];
        P[1] = ~A[2][0] & A[0][0];
        P[2] = ~A[0][0] & A[1][0];
        A[0][0] ^= P[0];
        A[1][0] ^= P[1];
        A[2][0] ^= P[2];

        P[0] = ~A[1][1] & A[2][1];
        P[1] = ~A[2][1] & A[0][1];
        P[2] = ~A[0][1] & A[1][1];
        A[0][1] ^= P[0];
        A[1][1] ^= P[1];
        A[2][1] ^= P[2];

        P[0] = ~A[1][2] & A[2][2];
        P[1] = ~A[2][2] & A[0][2];
        P[2] = ~A[0][2] & A[1][2];
        A[0][2] ^= P[0];
        A[1][2] ^= P[1];
        A[2][2] ^= P[2];

        P[0] = ~A[1][3] & A[2][3];
        P[1] = ~A[2][3] & A[0][3];
        P[2] = ~A[0][3] & A[1][3];
        A[0][3] ^= P[0];
        A[1][3] ^= P[1];
        A[2][3] ^= P[2];

        /* rho-east */
        A[1][0] = rotate_left32(A[1][0], 1);
        A[1][1] = rotate_left32(A[1][1], 1);
        A[1][2] = rotate_left32(A[1][2], 1);
        A[1][3] = rotate_left32(A[1][3], 1);

        P[0] = A[2][0];
        A[2][0] = rotate_left32(A[2][2], 8);
        A[2][2] = rotate_left32(P[0], 8);
        P[0] = A[2][1];
        A[2][1] = rotate_left32(A[2][3], 8);
        A[2][3] = rotate_left32(P[0], 8);
    }
}

#endif
