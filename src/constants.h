/**
 * *****************************************************************************
 * \file constants.h
 * \author Graham Beck
 * \brief BPPR: Constants, largely for facilitating BLAS/LAPACK operations. 
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include <filesystem>

#include "cx_math.h"
#include "mkl_vsl.h"


namespace bppr
{
    static constexpr float IGOLDENRATIO = 0.618034;
    static constexpr float KEPLER = 0.114942;
    static constexpr float IPI = 0.318310;
    static constexpr float WYLER = 0.007297;

    static constexpr MKL_INT SINGLESTEP    = 1;
    static constexpr MKL_INT NOSTEP            = 0;
    static constexpr MKL_INT ORIENTATION = VSL_SS_MATRIX_STORAGE_ROWS;
    static constexpr MKL_INT QPTYPE            = VSL_SS_SQUANTS_ZW_PARAMS_N;
    static constexpr float ONEf                         = 1; 
    static constexpr float NEGATIVEONEf       = -1; 
    static constexpr float ZEROf                        = 0; 
    static constexpr char TRANSPOSED           = 'T'; 
    static constexpr char UNTRANSPOSED     = 'N'; 
    static constexpr char UPPERTRIANGLE     = 'U'; 
    static constexpr char LOWERTRIANGLE    = 'L'; 
    static constexpr char NONUNITARY           = 'N'; 
    static constexpr char LEFTSIDE                   = 'L'; 
    static constexpr char RIGHTSIDE                = 'R'; 
   
    static constexpr char PATHSEP[2]               = {std::filesystem::path::preferred_separator, '\0'};
    static constexpr char SUFFIX[]                     = ".bppr"; 
    static constexpr char UPDTF[]                     = ".updt"; 
    static constexpr char L3FILE[]                      = ".l3b"; 

    static constexpr float SPHERICALMU[3][3] = {{1, 0, 0}, {1/cx::sqrt(2), 1/cx::sqrt(2), 0}, {1/cx::sqrt(3), 1/cx::sqrt(3), 1/cx::sqrt(3)}};

} // namespace bppr

// Satisfy the extern error messages in cx::err namespace
namespace cx
{
  namespace err
  {
    namespace
    {
      const char* abs_runtime_error = "Abs Runtime error";
      const char* sqrt_domain_error = "Sqrt Domain error";
      const char* exp_runtime_error = "Exp Runtime error";
      const char* floor_runtime_error= "Floor Runtime error";
      const char* ceil_runtime_error= "Ceil Runtime error";
      const char* fmod_domain_error= "Fmod Runtime error";
      const char* remainder_domain_error= "Remainder Domain error";
      const char* log_domain_error= "Log Domain error";
      const char* pow_runtime_error= "Pow Runtime error";
    }
  } // namespace err

} // namespace cx