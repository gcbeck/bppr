/**
 * *****************************************************************************
 * \file util.h
 * \author Graham Beck
 * \brief BPPR: Useful & rather random utilities. 
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <cstddef>

#include "cx_math.h"
#include "mkl_types.h"

template <int... N> struct RawQ;

template <int I, int N>
struct RawQ<I, N>
{
    static void run(float (&rQuantiles)[N], const float& zerothQ)
    {
        rQuantiles[I] = zerothQ + ((1 - 2 * zerothQ) / N) * I;
        RawQ<I + 1, N>::run(rQuantiles, zerothQ);
    }
};
template <int N>
struct RawQ<N, N>
{
    static void run(float (&rQuantiles)[N], const float& zerothQ) {}
};


/**
* @brief Compile-time calculation of the quantiles partitioning the interval 
*              [zerothQ, 1-zerothQ] into N equal subintervals for use as knots
*
* @details The float[N] array returned does not include the highest (N+1'th) 
*                  quantile as it is unnecessary for the spline basis
*                    
* @example RawQ<6>().run(0.02) -> [0.02, 0.18, 0.34, 0.50, 0.66, 0.82]
*/
template <int N>
struct RawQ<N>
{
    const float(&run(const float& zerothQ))[N]
    {
        RawQ<0, N>::run(_rQuantiles, zerothQ);
        return _rQuantiles;
    }

    float _rQuantiles[N];
};

/**
* @brief Log of the n-choose-r combination
*/
constexpr float lnCr(const unsigned int n, unsigned int r) {
    float x = 0;
    r = n - r > r ? n - r : r;
    for (unsigned int ix = 1; ix <= n - r; ++ix) { x += static_cast<float>(std::log(r + ix)); x -= static_cast<float>(std::log(ix)); }
    return x;
}
namespace cx
{
/**
* @brief Compile-time version of the log of the n-choose-r combination
*
* @note Better to use the runtime version at runtime, as  cx::log() involves deep recursion. 
*/
constexpr float lnCr(const unsigned int n, unsigned int r) {
    float x = 0;
    r = n - r > r ? n - r : r;
    for (unsigned int ix = 1; ix <= n - r; ++ix) { x += cx::log(r + ix); x -= cx::log(ix); }
    return x;
  }
} // namespace cx

template<size_t T>
constexpr unsigned short nchar() {
    unsigned short l = 0;
    for (auto n = T; n; l++, n /= 10);
    return l;
}

/**
* @brief Compile-time concatenation of multiple raw char arrays
*              into a std::array<char, .> 
*/
template<unsigned ...L>
constexpr auto join(const char (&...strings)[L]) {
  constexpr unsigned short N = (... + L) - sizeof...(L);
  std::array<char, N + 1> joined = {};
  joined[N] = '\0';

  auto it = joined.begin();
  (void)((it = std::copy_n(strings, L-1, it), 0), ...);
  return joined;
}

/**
* @brief Compile-time concatenation of a std::array<char, .> prefix, 
*               a integer T and a const char[.] postfix, useful in particular 
*               as a filepath for bppr files
*/
template<size_t T, size_t L, unsigned short U>
constexpr auto join(const std::array<char, L>& prefix, const char (&postfix)[U]) {
    constexpr unsigned short M = nchar<T>();
    constexpr unsigned short N = L + U + M - 1;
    std::array<char, N> joined = {};
    joined[N-1] = '\0';
    auto it = joined.begin();
    it = std::copy_n(prefix.begin(), L-1, it);
    it += M;
    for (auto n = T; n; n /= 10) {*--it = "0123456789"[(n % 10)]; }
    std::copy_n(postfix, U-1, it+M);
    return joined;
}

/**
* @brief Realigns circular/ring buffers that are col-major matrices
*              with U rows and L columns, given current head and tail indices. 
*
* @details Each segment entering or leaving the buffers has U rows and 
*                  runtime-dependent columns. The tail and head indices index 
*                  those columns. This function pulls all data back so that the 
*                  resulting tail index would be zero.  
*/
template<unsigned short U, unsigned short L>
void rectify(float (&circ)[U*L], const MKL_UINT& hx, const MKL_UINT& tx) {
    if (hx > tx) {
        std::memmove(circ, circ+tx*U, (hx > tx)*U*sizeof(float));
    } else if (2*tx >= L + hx) {
        std::memmove(circ+U*(L-tx), circ, hx*U*sizeof(float));
        std::memcpy(circ, circ+U*tx, U*(L-tx)*sizeof(float));
    } else {
        float* buffer = new float[hx*U];
        std::memcpy(buffer, circ, hx*U*sizeof(float));
        std::memmove(circ, circ+U*tx, U*(L-tx)*sizeof(float));
        std::memcpy(circ+U*(L-tx), buffer, hx*U*sizeof(float));
        delete[] buffer;
    }
}
