/**
 * *****************************************************************************
 * \file logging.h
 * \author Graham Beck
 * \brief BPPR: Utilize the lightweight logging library L3 for writing both integers & floats. 
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

// Define Logging Levels. Higher -> More Informative
#define ERRONEOUS 1
#define LACONIC 2
#define VERBOSE 3

#ifndef LOGLEVEL
#define LOGLEVEL LACONIC
#endif

#if LOGLEVEL >= ERRONEOUS
#define LOGERR(msg, prec, ...) log<prec>("ERROR : " msg, __VA_ARGS__)
#else
#define LOGERR(msg, ...) 
#endif

#if LOGLEVEL >= LACONIC
#define LOGLAC(msg, prec, ...) log<prec>("INFO  : " msg, __VA_ARGS__)
#else
#define LOGLAC(msg, ...) 
#endif

#if LOGLEVEL >= VERBOSE
#define LOGVER(msg, prec, ...) log<prec>("DEBUG : " msg, __VA_ARGS__)
#else
#define LOGVER(msg, ...)
#endif

#define APPEND(LVL, fmt, prec, ...)  \
    if constexpr(LOGLEVEL >= LVL) { log<prec>("------> " fmt, __VA_ARGS__); }

#include <cmath>

#include "l3.h"


using prec_t = uint64_t; // The precision desired for float arguments to l3 logging
namespace prec {
    static constexpr prec_t _       = 0x0000000000000001;
    static constexpr prec_t _X     = 0x0000000000000000;
    static constexpr prec_t _XX   = 0x8000000000000000;
    static constexpr prec_t _XXX = 0xC000000000000000;

    template<prec_t P> float multiplier() { return 10.0f; }
    template<> float multiplier<_XX>() { return 100.0f; }
    template<> float multiplier<_XXX>() { return 1000.0f; }

   /**
    * @brief Transform the decimal part of an l3 float to a representable uint64_t
    *
    * @details Contructs an integer representing the fractional part, ensures that rounding
    *                  does not take it out of scope, and encodes a significant-bit pattern to inform 
    *                  l3_dump.py of the precision required. 
    */
    template<prec_t P> 
    uint64_t enc(const float& v) { 
        return P | static_cast<uint64_t>(std::min(std::round(std::fabsf(v-std::truncf(v))*multiplier<P>()), multiplier<P>()-1)); 
    }
}

/**
* @brief The l3 log function that takes two integers or one float
*
* @details By default, takes two integers with respective specifiers in the msg. 
*                  Specifiers can be %u : unsigned int (though this will be bludgeoned into a %d)
*                                                  %n: negative or positive int but special care taken to observe negativity
*                                                  %d: generic int, take potluck as to whether negativity will be preserved. 
* @example log<prec::_>("Int1 %u and Int2 %n", 3, -2)
*
*                  The specialized-precision forms are for a single float input. This should always be represented
*                  with %n.%u (special care taken of negativity) or %u.%u (positive float expected)
* @example log<prec::_XX>("Negative float to two decimal places %n.%u", -3.14)
*
* @note If any more variation is required, consider wrapping in a partially-specializable struct 
*              so eg. prec::enc<P>(v) can be used rather than each specialization made explicit
*/
template<prec_t P, typename... U>
void log(const char* msg, const U&... vs) {
    l3_log_fn(msg, static_cast<uint64_t>(vs < 0 ? vs-1 : vs)..., L3_ARG_UNUSED);
}
template<> void log<prec::_X, float>(const char* msg, const float& v) {
    l3_log_fn(msg, static_cast<uint64_t>(std::floor(v)), prec::enc<prec::_X>(v), L3_ARG_UNUSED);
}
template<> void log<prec::_XX, float>(const char* msg, const float& v) {
    l3_log_fn(msg, static_cast<uint64_t>(std::floor(v)), prec::enc<prec::_XX>(v), L3_ARG_UNUSED);
}
template<> void log<prec::_XXX, float>(const char* msg, const float& v) {
    l3_log_fn(msg, static_cast<uint64_t>(std::floor(v)), prec::enc<prec::_XXX>(v), L3_ARG_UNUSED);
}