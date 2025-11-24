/**
 * *****************************************************************************
 * \file projection.h
 * \author Graham Beck
 * \brief BPPR: Linearly tranforms the raw training data matrix X into ridge func projections
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include "descriptor.h"
#include "mkl_blas.h"
#include "proto.h"
#include "util.h"


 namespace bppr
{
    template <bppr_t T>
    class Projection
    {
      public:
        static constexpr MKL_INT N = bpprix::decode<bpprix::kN>(T);
        /**
         * @brief Projection primarily performs the linear transformation X*theta 
         *              where X is a column-major matrix containing N samples of M signals 
         *              and theta is the matrix constituting nRidge columns of M coefficients.
         *              The constructor takes a Descriptor & Proto for initialization. 
         */
        Projection(const Descriptor<T>& dscr, Proto<T>& proto)
        {
            std::memset(_transform, 0, M*R*sizeof(float));
            std::memset(_index, 0, R*sizeof(size_t));
            proto.template get<proto::kProj>(_X, _transform, _cache);
            index();
        }

      /**
        * @brief The on-the-fly (rather than cached) projection for a single ridge function
        */
        void get(const unsigned short rix, float(&out)[N]) const {
            sgemv(&UNTRANSPOSED, &N, &M, &ONEf, _X, &N, _transform+rix*M, &SINGLESTEP, &ZEROf, out, &SINGLESTEP);
        }

      /**
        * @brief The fast form of get for prediction, where a single-sample regressor X gives up to R*P projections
        * 
        * @note  Easily generalized to multiple regressing samples. 
        */
        void get(const MKL_UINT(&rcx)[rcx::N], const float(&X)[bpprix::decode<bpprix::kM>(T)], float(&out)[bpprix::decode<bpprix::kR>(T)*bpprix::decode<bpprix::kP>(T)]) const {
            const MKL_INT n = (rcx[rcx::kPHx] <= rcx[rcx::kPTx] ? PR : rcx[rcx::kPHx]) - rcx[rcx::kPTx];
            sgemv(&TRANSPOSED, &M, &n, &ONEf, _cache+rcx[rcx::kPTx]*M, &M, X, &SINGLESTEP, &ZEROf, out, &SINGLESTEP);
            // Wrap around if necessary
            if (rcx[rcx::kPHx] <= rcx[rcx::kPTx]) {
                sgemv(&TRANSPOSED, &M, reinterpret_cast<const MKL_INT*>(&rcx[rcx::kPHx]), &ONEf, _cache, &M, X, &SINGLESTEP, &ZEROf, out+n, &SINGLESTEP);
            }
        }

      /**
        * @brief The  form of get for online updates, where a single-sample regressor X gives up to R projections 
        *               and may be inserted into _X at index isx. 
        * 
        * @note  Easily generalized to multiple regressing samples. 
        */
        void get(const MKL_INT& nRidge, const float(&X)[bpprix::decode<bpprix::kM>(T)], float(&out)[bpprix::decode<bpprix::kR>(T)]) {
            sgemv(&TRANSPOSED, &M, &nRidge, &ONEf, _transform, &M, X, &SINGLESTEP, &ZEROf, out, &SINGLESTEP);
        }
        void get(const unsigned short isx, const MKL_INT& nRidge, const float(&X)[bpprix::decode<bpprix::kM>(T)], float(&out)[bpprix::decode<bpprix::kR>(T)]) {
            get(nRidge, X, out); 
            scopy(&M, X, &SINGLESTEP, _X+isx, &N);
        }

      /**
        * @brief The canonical get for the class, returning its main feature: the transform coefficients
        */
        const float(&get())[bpprix::decode<bpprix::kM>(T) * bpprix::decode<bpprix::kR>(T)] {
            return  _transform;
        }

      /**
        * @brief Sets a new set of coefficients at (ridge-) index rix. The inputs are an array of the signal indices 
        *              and an array of the respective values, which should lie on a hypersphere of dimension U. 
        * 
        * @details Most often used to construct a new ridge transformation adjacent to the current ones. 
        *                  Also updates the signal-index tracking so the signals used by each ridge can be found quickly.   
        */
        template<unsigned short U>
        void set(const unsigned short rix, const unsigned short(&mix)[U], const float(&in)[U], float(&out)[N]) {
            std::memset(_transform+rix*M, 0, M*sizeof(float));
            std::memset(out, 0, N*sizeof(float));
            _index[rix] = 0;
            for (unsigned short ix = 0; ix < U; ++ix) {
                _transform[rix*M + mix[ix]] = in[ix];
                _index[rix] |= (1 << mix[ix]);
                saxpy(&N, in+ix, _X+N*mix[ix], &SINGLESTEP, out, &SINGLESTEP);
            }
        }
      /**
        * @brief The form of set used when the ridge function constitutes a single signal. 
        */
        void set(const unsigned short rix, const unsigned short mix, const bool negated, float(&out)[N]) {
            std::memset(_transform+rix*M, 0, M*sizeof(float));
            _index[rix] = (1 << mix);
            if (negated) {
                std::memset(out, 0, N*sizeof(float));
                _transform[rix*M + mix] = -1;
                saxpy(&N, _transform+rix*M+mix, _X+N*mix, &SINGLESTEP, out, &SINGLESTEP);
            } else {
                _transform[rix*M + mix] = 1;
                std::memcpy(out, _X+N*mix, N*sizeof(float));
            }
        }

      /**
        * @brief Removes a ridge slot, pulling all subsequent ridge coefficients into the gap. 
        */
        void del(const unsigned short rix, const unsigned short nRidge) {
            std::memmove(_transform+rix*M, _transform+(rix+1)*M, M*(nRidge-rix-1)*sizeof(float));
            std::memmove(_index+rix, _index+rix+1, (nRidge-rix-1)*sizeof(size_t));
        }

      /**
        * @brief Used by an accepted 'change' action, the proposed transformation at index nRidge
        *              overwrites the previous one at rix.  
        */
        void mod(const unsigned short rix, const unsigned short nRidge) {
            std::memcpy(_transform+rix*M, _transform+nRidge*M, M*sizeof(float));
        }

      /**
        * @brief Copies the transformation matrix into a circular buffer and updates its head and tail  
        */
        void cache(MKL_UINT(&rcx)[rcx::N], const unsigned short nRIn, const unsigned short nROut) {
            unsigned short n = std::min(static_cast<MKL_UINT>(nRIn), PR-rcx[rcx::kPHx]);
            std::memcpy(_cache+rcx[rcx::kPHx]*M, _transform, n*M*sizeof(float));

            // Wrap around if necessary
            n = nRIn - n;
            if (n > 0) { std::memcpy(_cache, _transform+M*(nRIn-n), n*M*sizeof(float)); }

            rcx[rcx::kPHx] += nRIn; rcx[rcx::kPHx] %= PR;
            if (rcx[rcx::kPn] >= bpprix::decode<bpprix::kP>(T)) {
                rcx[rcx::kPTx] += nROut; rcx[rcx::kPTx] %= PR;
            }
        }

      /**
        * @brief Computes the cosine measures of similarity between the transformation at rix and the n transforms starting at rsx 
        */
        void cos(const MKL_INT rsx, const MKL_INT n, const MKL_INT rix, float* out, const MKL_INT incr, const float beta) {
            sgemv(&TRANSPOSED, &M, &n, &ONEf, _transform+rsx*M, &M, _transform+rix*M, &SINGLESTEP, &beta, out, &incr);
        }

      /**
        * @brief Runs through all possible ridge functions, computing the tracking index for each.
        *              Returns the number of ridge functions. 
        */
        unsigned short index() {
            unsigned short nix = 0;
            for (unsigned short rix = 0; rix < R; ++rix) {
                for (unsigned short mix = 0; mix < M; ++mix) {
                    if (_transform[rix*M + mix] != 0) { _index[rix] |= (1 << mix); }
                }
                nix += (_index[rix] > 0);
            }
            return nix;
        }
      /**
        * @brief Returns the number of signals used by the ridge at index rix. 
        */
        unsigned short index(const unsigned short& rix) {
            return __builtin_popcount(_index[rix]);
        }
      /**
        * @brief Populates an array of the 1-based signal indices used and the respective coefficients at ridge rix. 
        *              Returns the number of signals used, though that would normally already be known. 
        */
        template<typename... U>
        unsigned short index(const unsigned short& rix, U(&...pm)[bpprix::decode<bpprix::kA>(T)]) {
            size_t rindex = _index[rix];
            unsigned short nix = 0;
            for (unsigned short ix = 0; ix < M; ++ix) { 
                if ((rindex >> ix) & 1) { 
                    (pmx<U>(rix, ix, pm[nix]), ...); // pmix needs to be 1-indexed before entering dwallenius
                    ++nix;
                    rindex &= ~(1 << ix);
                }
                if (rindex == 0) { break; }
            }
            return nix;
        }

      /**
        * @brief Tests if any ridge functions already constitute the signals encoded in 'test', at least
        *              'multiplicity' times over all ridge functions. 
        */
        bool has(const unsigned short test, const unsigned short nRidge, unsigned short multiplicity) {
            for (unsigned short ix = 0; ix < nRidge; ++ix) { 
                if ((_index[ix] == test) && (--multiplicity == 0)) {
                    return true;
                }
            }
            return false;
        }

      /**
        * @brief Persist state to file. 
        */
        void write(Proto<T>& proto, const unsigned short nRidge, MKL_UINT(&rcx)[rcx::N]) {
            rectify<M, PR>(_cache, rcx[rcx::kPHx], rcx[rcx::kPTx]);
            rcx[rcx::kPTx] = 0; rcx[rcx::kPHx] = rcx[rcx::kRx];
            proto.template set<proto::kProj>(nRidge, _X, _transform, rcx[rcx::kRx], _cache);
        }

      private:
        static constexpr MKL_INT M            = bpprix::decode<bpprix::kM>(T); // Number of coefficients for each projection = number of signals
        static constexpr unsigned short R  = bpprix::decode<bpprix::kR>(T);  // Number of ridge functions = number of projections
        static constexpr MKL_INT PR           = bpprix::decode<bpprix::kP>(T)*R;

        template<typename U>
        void pmx(const unsigned short rix, const unsigned short ix, U& out) {}
        template<> void pmx<unsigned short>(const unsigned short rix, const unsigned short ix, unsigned short& out) { out = ix + 1; } // pmix needs to be 1-indexed before entering dwallenius
        template<> void pmx<float>(const unsigned short rix, const unsigned short ix, float& out) { out = _transform[rix*M+ix]; }

        float _X[N * M];                 // The signals matrix in col-major ordering
        float _transform[M * R]; // Coefficients linearly transforming signals to projections. [Ridge1Coeffs, Ridge2Coeffs, ...]
        float _cache[PR*M];        // Circular buffer of up to P _transforms. 
        size_t _index[R];               // An encoding of the signals used by each ridge function. 
    };

} // namespace bppr