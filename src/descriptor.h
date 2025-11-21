/**
 * *****************************************************************************
 * \file descriptor.h
 * \author Graham Beck
 * \brief BPPR: Provides the encoding for the compile-time parameters 
 *                        and the container for the runtime parameters
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

#include "constants.h"
#include "mkl_types.h"


namespace bppr
{
    /**
     * @brief Returns in function 'rate' the mean rate for a Poisson process that would return the number of events
     *              'U' or higher with probability InverseGoldenRatio^8 = 0.021286656.
     * 
     * @example PoissonRate<8>::rate() -> 3.35
     */
    template <unsigned short U>
    struct PoissonRate
    {
        using ix_t = unsigned short;
        static constexpr const float kRates[] = {1.04, 1.22, 1.44, 1.68, 1.96, 2.27, 2.62, 3.02, 3.46, 3.96, 4.52, 5.14,
                                           5.84, 6.62, 7.48, 8.44, 9.51, 10.69, 12.01, 13.46, 15.08, 16.86, 18.83,
                                           21.01, 23.42, 26.08, 29.01, 32.25, 35.81, 39.74, 44.07, 48.83};
        static constexpr const float kMinExponent = 2;
        static constexpr const float kIncrExponent = 0.129032258;

        static inline constexpr auto rate = []()
        {
            constexpr const float* pRates = kRates;  // Workaround for non-bounded array reference in constexpr issue. 
            constexpr auto ix = (cx::log2(static_cast<float>(U)) - kMinExponent) / kIncrExponent;
            if constexpr (ix <= 0) {
                return pRates[0];
            }
            if constexpr (ix < (sizeof(kRates) / sizeof(float) - 1)) {
                return pRates[static_cast<ix_t>(cx::ceil(ix))] * (ix - cx::floor(ix)) + kRates[static_cast<ix_t>(cx::floor(ix))] * (cx::ceil(ix) - ix);
            }
            return pRates[static_cast<ix_t>(sizeof(kRates)) / sizeof(float) - 1];
        };

    };

    using bppr_t = size_t;
    using bpprix_t = unsigned short;

    namespace bpprix
    {
        static const bpprix_t kN = 0x0000;
        static const bppr_t kNMask = 0x000000000000FFFF; // N has max 65535
        static const bpprix_t kNShift = 0;

        static const bpprix_t kM = 0x0001;
        static const bppr_t kMMask = 0x00000000007F0000; // M has max 127
        static const bpprix_t kMShift = 16;

        static const bpprix_t kR = 0x0002;
        static const bppr_t kRMask = 0x000000003F800000; // R has max 127
        static const bpprix_t kRShift = 23;

        static const bpprix_t kA = 0x0004;
        static const bppr_t kAMask = 0x00000001C0000000; // A has max 8
        static const bpprix_t kAShift = 30;

        static const bpprix_t kK = 0x0008;
        static const bppr_t kKMask = 0x0000007E00000000; // K has max 63
        static const bpprix_t kKShift = 33;

        static const bpprix_t kP = 0x0010;
        static const bppr_t kPMask = 0x007FFF8000000000; // P has max 65535
        static const bpprix_t kPShift = 39;

        static const bpprix_t kI = 0x0020;
        static const bppr_t kIMask = 0x0080000000000000; // I has max 1
        static const bpprix_t kIShift = 55;

        // static const bpprix_t kNext = 0x0040;
        // static const bpprix_t kNextShift = 56;

        static const bppr_t kInvalid = 0;

        template <bpprix_t T>
        static constexpr bppr_t decode(const bppr_t& type) { return kInvalid; }

        template <>
        constexpr bppr_t decode<kN>(const bppr_t& type) { return (type & kNMask) >> kNShift; }
        template <>
        constexpr bppr_t decode<kM>(const bppr_t& type) { return (type & kMMask) >> kMShift; }
        template <>
        constexpr bppr_t decode<kR>(const bppr_t& type) { return (type & kRMask) >> kRShift; }
        template <>
        constexpr bppr_t decode<kA>(const bppr_t & type) { return (type & kAMask) >> kAShift; }
        template <>
        constexpr bppr_t decode<kK>(const bppr_t& type) { return (type & kKMask) >> kKShift; }
        template <>
        constexpr bppr_t decode<kP>(const bppr_t& type) { return (type & kPMask) >> kPShift; }
        template <>
        constexpr bppr_t decode<kI>(const bppr_t& type) { return (type & kIMask) >> kIShift; }

        template <bpprix_t T>
        static constexpr bppr_t encode(const bppr_t& type) { return kInvalid; }

        template <>
        constexpr bppr_t encode<kN>(const bppr_t& type)
        {
            assert(type <= (kNMask >> kNShift));
            return type << kNShift;
        }
        template <>
        constexpr bppr_t encode<kM>(const bppr_t& type)
        {
            assert(type <= (kMMask >> kMShift));
            return type << kMShift;
        }
        template <>
        constexpr bppr_t encode<kR>(const bppr_t& type)
        {
            assert(type <= (kRMask >> kRShift));
            return type << kRShift;
        }
        template <>
        constexpr bppr_t encode<kA>(const bppr_t& type)
        {
            assert(type <= (kAMask >> kAShift));
            return type << kAShift;
        }
        template <>
        constexpr bppr_t encode<kK>(const bppr_t& type)
        {
            assert(type <= (kKMask >> kKShift));
            return type << kKShift;
        }
        template <>
        constexpr bppr_t encode<kP>(const bppr_t& type)
        {
            assert(type <= (kPMask >> kPShift));
            return type << kPShift;
        }
        template <>
        constexpr bppr_t encode<kI>(const bppr_t& type)
        {
            assert(type <= (kIMask >> kIShift));
            return type << kIShift;
        }

        static constexpr bppr_t encode(MKL_UINT&& N, MKL_UINT&& M, unsigned short&& R, unsigned short&& A, unsigned short&& K, MKL_UINT&& P, bool&& I)
        {
            return encode<kN>(N) | encode<kM>(M) | encode<kR>(R) | encode<kA>(A) | encode<kK>(K) | encode<kP>(P) | encode<kI>(I);
        }

    } // namespace bpprix

    /**
     * @brief BPPR Parameterization
     */
    template <bppr_t T>
    class Descriptor
    {
    public:
        static constexpr MKL_UINT N = bpprix::decode<bpprix::kN>(T);       // Max number of samples
        static constexpr MKL_UINT M = bpprix::decode<bpprix::kM>(T);       // Number of signals
        static constexpr unsigned short R = bpprix::decode<bpprix::kR>(T); // Max number of ridge functions
        static constexpr unsigned short A = bpprix::decode<bpprix::kA>(T); // Max number of active signals in any given ridge function
        static constexpr unsigned short K = bpprix::decode<bpprix::kK>(T); // Number of Spline degrees of freedom
        static constexpr bool I = bpprix::decode<bpprix::kI>(T); // Whether to incorporate an intercept coefficient
        static constexpr MKL_UINT P = bpprix::decode<bpprix::kP>(T); // Number of posterior MCMC draws used to make predictions

        static constexpr float kLambda = PoissonRate<R>::rate(); // Mean Poisson rate for sampling ridge function number
        static constexpr unsigned int kSeed = 512; // Default random seed to initialize the Marsenne Twister engine

        const MKL_UINT nBurn;             // Number of draws to burn before obtaining P draws for inference
        const unsigned short nAdapt;  // Number of MCMC iterations before burn-in, skipping sampling basis coefficients & residual variance
        const unsigned short nEvery;   // Keep every nEvery'th posterior sample to make up the dictionary used for inference
        const MKL_UINT rSeed;             // The random seed that initializes the Marsenne Twister engine
        
        static constexpr float kKnotQuNScale = IPI;   // Standard deviation of the Normal prior for the quantile steepness parameter 'nu'
        static constexpr float kKnotQrZero = cx::pow(IGOLDENRATIO, 8); // The Zeroth raw (pre-sigmoidal) quantile value
        static constexpr float kBetaIGVShape = 0.5;   // Shape parameter for the Inverse-Gamma prior for the basis function variance
        static constexpr float kBetaIGVScale = N / 2; // Scale parameter for the Inverse-Gamma prior for the basis function variance
        static constexpr float kSphericalK = 16;          // 'Precision' parameter (kappa) of spherical distribution for sampling projections

        const float kActiveW[A];  // Weighting applied to the sampling of the active signal number
        const float kIndexW[M];  // Weighting applied to the sampling of the signals themselves

        // Extend the settable scope of this Descriptor by making just const any needed static constexpr float and setting it in the ctor
        Descriptor(const MKL_UINT&& nBurn, const unsigned short&& nAdapt, const unsigned short&& nEvery, const MKL_UINT&& rSeed, std::initializer_list<float>&& kActiveW, std::initializer_list<float>&& kIndexW)
            : Descriptor(std::move(nBurn), std::move(nAdapt), std::move(nEvery), std::move(rSeed)
            , std::move(kActiveW), std::make_index_sequence<A>{}, std::move(kIndexW), std::make_index_sequence<M>{}) {}
        Descriptor(const MKL_UINT&& nBurn, const unsigned short&& nAdapt, const unsigned short&& nEvery, const MKL_UINT&& rSeed)
            : Descriptor(std::move(nBurn), std::move(nAdapt), std::move(nEvery), std::move(rSeed)
            , []<std::size_t... Is>(std::index_sequence<Is...>)->std::initializer_list<float>{ return { (1.0f / std::sqrtf(Is+1))... }; }(std::make_index_sequence<A>{}), std::make_index_sequence<A>{}
            , []<std::size_t... Is>(std::index_sequence<Is...>)->std::initializer_list<float>{ return { (static_cast<void>(Is), 1.0f)... }; }(std::make_index_sequence<M>{}), std::make_index_sequence<M>{}) {}
        Descriptor(const MKL_UINT&& nBurn, const unsigned short&& nAdapt, const unsigned short&& nEvery)
            : Descriptor(std::move(nBurn), std::move(nAdapt), std::move(nEvery), static_cast<const MKL_UINT&&>(kSeed)) {}

      private:
        template<std::size_t... iA, std::size_t... iM>
        Descriptor(const MKL_UINT&& nBurn, const unsigned short&& nAdapt, const unsigned short&& nEvery, const MKL_UINT&& rSeed, 
                            std::initializer_list<float>&& kActiveW, std::index_sequence<iA...>, std::initializer_list<float>&& kIndexW, std::index_sequence<iM...>)
            : nBurn(nBurn), nAdapt(nAdapt), nEvery(nEvery), rSeed(rSeed)
            , kActiveW{(*(kActiveW.begin() + iA))...}, kIndexW{(*(kIndexW.begin() + iM))...}{}

        static_assert((K > 0) && (N >= K*R + I));
        static_assert(M < sizeof(MKL_UINT) * 8);
        static_assert(P <= N / R);
    };

} // namespace bppr