/**
 * *****************************************************************************
 * \file knots.h
 * \author Graham Beck
 * \brief BPPR: Performs the regression against the spline basis & maintains Bayesian terms. 
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include <algorithm>

#include "knots.h"
#include "mkl_lapack.h"


namespace bppr
{
    using action_t = unsigned short;
    namespace action {
        static const action_t kNone           = 0x0000; 
        static const action_t kBirth            = 0x0001; 
        static const action_t kChange       = 0x0002;
        static const action_t kDeath          = 0x0003; 

        static const action_t N                    = kDeath + 1; 
    } // namespace action

    using pred_t = unsigned short;
    namespace pred {
        static const pred_t  kUnsampled  = 0x0000; 
        static const pred_t  kMean            = 0x0001; 
        static const pred_t  kDistribution = 0x0002; 

        template<pred_t U>
        concept point_t = !(U & kDistribution);
    } // namespace pred

    using mhix_t = unsigned short;
    namespace mhix {
        static const mhix_t kSpageiria = 0; 
        static const mhix_t kProposal  = 1;
        static const mhix_t kSelection = 2;
        static const mhix_t kPrior         = 3;
        static const mhix_t kTotal         = 4;

        static const mhix_t N                 = kTotal + 1;
    } // namespace mhix

    using cholesky_t = unsigned short;
    namespace update {
        static const cholesky_t kFull           = 0x0000; 
        static const cholesky_t kRankOne = 0x0001; 
        static const cholesky_t kOdd          = 0x0002; // The (lower-triangular) matrix has odd size m implying an lda of m+2
        static const cholesky_t kEven         = 0x0004; // The (lower-triangular) matrix has even size m implying an lda of m+1
        static const cholesky_t kParity       = 0x0006; // Mask for determining if even or odd parity is set

        template<cholesky_t U>
        concept fullRank_t = !(U & update::kRankOne);
        template<cholesky_t U>
        concept rankOne_t = static_cast<bool>(U & update::kRankOne);
        template<cholesky_t U>
        concept paritized_t = rankOne_t<U> && static_cast<bool>(U & (update::kOdd | update::kEven));
        template<cholesky_t U>
        concept odd_t = fullRank_t<U> && static_cast<bool>(U & update::kOdd);
        template<cholesky_t U>
        concept even_t = fullRank_t<U> && static_cast<bool>(U & update::kEven);
    } // namespace update

    using factor_t = unsigned short;
    namespace factor {
        static const factor_t kNull          = 0x0000; 
        static const factor_t kBasis        = 0x0001; 
        static const factor_t kProj          = 0x0002; 

        static const factor_t kControl    = 0x0004; 
        static const factor_t kProposal = 0x0008; 
    } // namespace factor


    template<bppr_t T> class BPPR; // Forward declaration for access to ::state definitions

    template<bppr_t T>
    class ZellnerSiow
    {
      public:
      /**
        * @brief ZellnerSiow performs most of the heavy lifting: transforming the projections & knots into bases and
        *              testing proposal ridge functions against the current lineup by comparing the resulting regression sse 
        *              and anti-similarity factor along with other Bayesian terms. 
        * 
        * @details The two most important members are _basesky and _projesky. The former is a Factor - meaning a 
        *                  cholesky-factorization of positive definite matrix M^T*M for some M - of the spline basis matrix 
        *                  which is required to solve for the regression coefficients, while the latter is a Factor of the Projection 
        *                  transformation matrix which makes the calculation of the anti-similarity measure, a determinant, 
        *                  just a product of its diagonals.
        *                  Each Factor contains two Rectangular Full Packed Format (RFPF) triangular matrices - the control and the 
        *                  proposal. When a proposal is accepted, it becomes the control. 
        */
        ZellnerSiow(const Descriptor<T>& dscr, Proto<T>& proto)
            : _dscr(dscr)
            , _knots(dscr, proto)
            , _basesky(_basis)
            , _projesky(_knots.template get<Projection<T>>())
            , _tau(dscr.kBetaIGVScale / (dscr.kBetaIGVScale + dscr.kBetaIGVShape))
        {
            // Initialize and normalize the adaptive weight vectors here even if they are overwritten by proto
            _activeW[0] = 0;
            for (unsigned short ix = 0; ix < bpprix::decode<bpprix::kA>(T); ++ix) {
                _activeW[ix+1] = _activeW[ix] + dscr.kActiveW[ix];
            }
            _activeW[bpprix::decode<bpprix::kA>(T)+1] = _activeW[bpprix::decode<bpprix::kA>(T)];
            vsDivI(bpprix::decode<bpprix::kA>(T), _activeW+1, SINGLESTEP, _activeW+bpprix::decode<bpprix::kA>(T), NOSTEP, _activeW+1, SINGLESTEP);
            _indexW[0] = 0;
            for (unsigned short ix = 0; ix < bpprix::decode<bpprix::kM>(T); ++ix) {
                _indexW[ix+1] = _indexW[ix] + dscr.kIndexW[ix];
            }
            _indexW[bpprix::decode<bpprix::kM>(T)+1] = _indexW[bpprix::decode<bpprix::kM>(T)];
            vsDivI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, _indexW+bpprix::decode<bpprix::kM>(T), NOSTEP, _indexW+1, SINGLESTEP);

            unsigned short nRidge;
            const bool warmstart = proto.template get<proto::kZS>(nRidge, _y, _tau, _indexW, _activeW, _cache.first, _cache.second);
            _projesky.set(nRidge);
            update<update::kFull>(_projesky);
            _projesky.dset(); // Computes and caches the initial control determinant
            
            std::memset(_mh, 0, mhix::N*sizeof(float));
            if (bpprix::decode<bpprix::kI>(T)) {
                std::fill(_basis, _basis+N, 1); 
            }
            for (unsigned short rix = 0; rix < _projesky.k(); ++rix) {
                toBasis(rix, _knots.get(rix));
            }

             _basesky.set(_projesky.k()*(kD-2) + bpprix::decode<bpprix::kI>(T));
             _ssy = sdot(&N, _y, &SINGLESTEP, _y, &SINGLESTEP);
            update<update::kFull>(_basesky, _y, _by);
            _sxy = beta<action::kNone>();
            _sse = _ssy - _sxy;
        }

      /**
        * @brief Switchyard for the mcmc action and regression coefficient sampling/caching. 
        * 
        * @details A death cannot occur when there is only a single ridge function; likewise neither a birth nor change
        *                 may occur when R ridge functions are already established (in the case of a change proposal this is
        *                 only due to storage restrictions). 
        *                 In the Adapt phase the sse may improve with accepted ridge proposals but the shrinkage and variance 
        *                 hyperparameters tau and sigma are static. They are sampled in the Burn phase while in addition the 
        *                 sampled regression coefficients beta are cached for posterior prediction in the Posterior phase.
        */
        template<typename BPPR<T>::state_t S> void mcmc(MKL_UINT(&rcx)[rcx::N]) {}
        template<>
        void mcmc<BPPR<T>::kAdapt>(MKL_UINT(&rcx)[rcx::N]) {
            const action_t action = getAction(_projesky.k());
            switch (action) {
            case action::kBirth:
                LOGLAC("[ZellnerSiow::mcmc] Iteration (%u,%u) : BIRTH Action", prec::_, rcx[rcx::kMCx], rcx[rcx::kRKx]);
                propose<action::kBirth>();
                break;
            case action::kChange:
                LOGLAC("[ZellnerSiow::mcmc] Iteration (%u,%u) : CHANGE Action", prec::_, rcx[rcx::kMCx], rcx[rcx::kRKx]);
                propose<action::kChange>();
                break;
            case action::kDeath:
                LOGLAC("[ZellnerSiow::mcmc] Iteration (%u,%u) : DEATH Action", prec::_, rcx[rcx::kMCx], rcx[rcx::kRKx]);
                propose<action::kDeath>();
                break;
            }
        }
        template<>
        void mcmc<BPPR<T>::kBurn>(MKL_UINT(&rcx)[rcx::N]) {
            mcmc<BPPR<T>::kAdapt>(rcx);
            sample(rcx);
        }
        template<>
        void mcmc<BPPR<T>::kPost>(MKL_UINT(&rcx)[rcx::N]) {
            mcmc<BPPR<T>::kBurn>(rcx);
            cache(rcx);
        }

      /**
        * @brief Introduce a new sample (X,y) to update the BPPR state with. 
        * 
        * @details Updating the basis cholesky involves two rank-one updates: the roll-on of the new X and a roll-off of the old one. 
        *                  The update currently does not force immediate adaptation of the regression coefficient vector beta, rather 
        *                  relying on changes to the sse or newly accepted proposed ridge functions to precipitate the recalculation. 
        */
        void update(const float(&X)[bpprix::decode<bpprix::kM>(T)], const float y, MKL_UINT(&rcx)[rcx::N]) {
            // Compute the new basis sample and update the cholesky factor
            toBasis(_projesky.k(), _knots.get(rcx[rcx::kNx], _projesky.k(), X), reinterpret_cast<float(&)[RK]>(_prsis[bpprix::decode<bpprix::kI>(T) ? 1 : 0]));
            if (bpprix::decode<bpprix::kI>(T)) { _prsis[0] = 1; }
            LOGVER("[ZellnerSiow::update] Supervised Y %n.%u", prec::_XX, y);
            APPEND(VERBOSE, "<-> Unsampled Pred Y %n.%u", prec::_XX, sdot(&_basesky.k(), _beta, &SINGLESTEP, _prsis, &SINGLESTEP));
            scopy(&_basesky.k(), _basis+rcx[rcx::kNx], &N, _prsis+_basesky.k(), &SINGLESTEP);
            scopy(&_basesky.k(), _prsis, &SINGLESTEP, _basis+rcx[rcx::kNx], &N);
            update<update::kRankOne, factor::kControl>(_basesky, reinterpret_cast<float(&)[Factor<factor::kBasis>::kM]>(_prsis), ONEf, ONEf);
            update<update::kRankOne, factor::kControl>(_basesky, reinterpret_cast<float(&)[Factor<factor::kBasis>::kM]>(_prsis[_basesky.k()]), ONEf, NEGATIVEONEf);

            // Compute the new basis response and update _by
            float ysqi = _y[rcx[rcx::kNx]]; ysqi *= -ysqi; ysqi += y*y;
            _ssy += ysqi;
            saxpy(&_basesky.k(), &y, _prsis, &SINGLESTEP, _by, &SINGLESTEP);
            _y[rcx[rcx::kNx]] = -_y[rcx[rcx::kNx]];
            saxpy(&_basesky.k(), &_y[rcx[rcx::kNx]], _prsis+_basesky.k(), &SINGLESTEP, _by, &SINGLESTEP);
            _y[rcx[rcx::kNx]] = y;

            if (++rcx[rcx::kNx]==N) {
                rcx[rcx::kNx] = 0;
            }
        }

      /**
        * @brief Poterior prediction using the full cache of P bppr samples. 
        *              Puts the P predictions into vector prsis for access by posterior(), and returns their mean. 
        */
        template<pred_t U>
        float predict(MKL_UINT(&rcx)[rcx::N], const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            toBasis<P>(rcx, _knots.get(rcx, X), _prsis);
            const float* beta = _cache.second + rcx[rcx::kZTx];
            const float* const bend = _cache.second + PRK;
            const float* basx = _prsis;
            float pred = 0;
            for (unsigned short ix = 0; ix < P; ++ix) {
                const MKL_INT m = _cache.first[(rcx[rcx::kPx]+ix) % P]*(kD-2);
                const MKL_INT n = beta + m < bend ? m : (bend-beta); 
                _prsis[ix] = sdot(&n, beta, &SINGLESTEP, basx, &SINGLESTEP);
                if (n < m) { 
                    const MKL_INT r = m - n;
                    _prsis[ix] += sdot(&r, _cache.second, &SINGLESTEP, basx+n, &SINGLESTEP);
                    beta = _cache.second + r;
                } else {
                    beta += n;
                }
                if constexpr(bpprix::decode<bpprix::kI>(T)) {
                    _prsis[ix] += _cache.second[(rcx[rcx::kPx]+ix) % P]; // The intercept term
                }
                basx += n;
                pred += _prsis[ix];
            }
            return pred / P;
        }
      /**
        * @brief The form of predict targeting more efficiently just the mean of the P samples.
        */
        template<>
        float predict<pred::kMean>(MKL_UINT(&rcx)[rcx::N], const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            toBasis<P>(rcx, _knots.get(rcx, X), _prsis);
            const MKL_INT n = (rcx[rcx::kZHx] > rcx[rcx::kZTx] ? rcx[rcx::kZHx] : PRK) - rcx[rcx::kZTx];
            float y = sdot(&n, _cache.second+rcx[rcx::kZTx], &SINGLESTEP, _prsis, &SINGLESTEP);
            if constexpr(bpprix::decode<bpprix::kI>(T)) {
                if (rcx[rcx::kZHx] <= rcx[rcx::kZTx]) {
                    const MKL_INT m = rcx[rcx::kZHx] - P;
                    y += sdot(&m, _cache.second+P, &SINGLESTEP, _prsis+n, &SINGLESTEP);
                }
                // In this case the first col of _basis is an array of ones; use that to sum the first P values in cache.second. 
                return (y + sdot(&P, _cache.second, &SINGLESTEP, _basis, &SINGLESTEP)) / P;
            } else if (rcx[rcx::kZHx] <= rcx[rcx::kZTx]) {
                    y += sdot(reinterpret_cast<MKL_INT*>(&rcx[rcx::kZHx]), _cache.second, &SINGLESTEP, _prsis+n, &SINGLESTEP);
            }
            return y / P;
        }
      /**
        * @brief The form of predict based on the current regression coefficients beta rather than the sampled cache of coefficients. 
        *              Usually the fastest but least robust prediction calculation. 
        */
        template<>
        float predict<pred::kUnsampled>(MKL_UINT(&rcx)[rcx::N], const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            toBasis(_projesky.k(), _knots.get(_projesky.k(), X), reinterpret_cast<float(&)[RK]>(_prsis[bpprix::decode<bpprix::kI>(T) ? 1 : 0]));
            if (bpprix::decode<bpprix::kI>(T)) { _prsis[0] = 1; }
            return sdot(&_basesky.k(), _beta, &SINGLESTEP, _prsis, &SINGLESTEP);
        }

      /**
        * @brief Returns the posterior array of prediction samples from which a distribution can be inferred. 
        */
        const float(&posterior())[bpprix::decode<bpprix::kP>(T)] {
            return reinterpret_cast<float(&)[P]>(_prsis);
        }
      /**
        * @brief Returns the Absolute Deviation array |posterior - centre| for use in spread or mean-AD calculations.
        * 
        * @note The posterior samples themselves are altered via this call, so posterior() will not subsequently 
        *              return the samples themelves. 
        */
        const float(&posterior(const float& centre))[bpprix::decode<bpprix::kP>(T)] { // ADs
            vsSubI(P, _prsis, SINGLESTEP, &centre, NOSTEP, _prsis, SINGLESTEP);
            vsAbs(P, _prsis, _prsis);
            return reinterpret_cast<float(&)[P]>(_prsis);
        }

      /**
        * @brief Logs the full projection coefficient vector for ridge ix or all ridge functions
        */
        void coefs(const unsigned short ix) {
            const unsigned short nActive = _knots.index(ix, _pmix, _pmu);
            LOGLAC("[ZellnerSiow::coefs] Ridge %u (nActive %u) : ", prec::_, ix, nActive); 
            for (unsigned short jx=0, kx=0; jx < bpprix::decode<bpprix::kM>(T); ++jx) {
                if (jx == _pmix[kx]-1) { 
                    APPEND(LACONIC, "%n.%u", prec::_XX, _pmu[kx++]);
                } else { 
                    APPEND(LACONIC, "%u", prec::_, 0, L3_ARG_UNUSED);
                }
            }
        }
        void coefs() {
            for (unsigned short ix = 0; ix < _projesky.k(); ++ix) { coefs(ix); }
        }

      /**
        * @brief Persist state to file. 
        */
        const unsigned short write(Proto<T>& proto, MKL_UINT(&rcx)[rcx::N]) {
            _knots.write(proto, _projesky.k(), rcx);
            const unsigned short p = bpprix::decode<bpprix::kI>(T) ? P : 0;  
            rectify<1, PR*(kD-2)>(reinterpret_cast<float(&)[PR*(kD-2)]>(_cache.second[p]), rcx[rcx::kKHx]-p, rcx[rcx::kKTx]-p);
            std::memcpy(_working, _cache.first, rcx[rcx::kPx]*sizeof(unsigned short));
            const unsigned short n = P-rcx[rcx::kPx];
            std::memmove(_cache.first, _cache.first+rcx[rcx::kPx], n*sizeof(unsigned short));
            std::memcpy(_cache.first+n, _working, rcx[rcx::kPx]*sizeof(unsigned short));
            rcx[rcx::kPx] = 0;
            rcx[rcx::kZTx] = p; rcx[rcx::kZHx] = rcx[rcx::kRKx]+p;
            proto.template set<proto::kZS>(_projesky.k(), _y, _tau, _indexW, _activeW, rcx[rcx::kRx], _cache.first, _cache.second);
            return _projesky.k();
        }

      private:
        static constexpr unsigned short RK = bpprix::decode<bpprix::kR>(T) * bpprix::decode<bpprix::kK>(T) 
                                                                           + bpprix::decode<bpprix::kI>(T) ; // Max number of bases over all ridge functions
        static constexpr MKL_INT N               = bpprix::decode<bpprix::kN>(T);   // Number of data samples
        static constexpr unsigned short NR = N * RK;
        static constexpr unsigned short MR = bpprix::decode<bpprix::kM>(T) * bpprix::decode<bpprix::kR>(T) ;
        static constexpr unsigned short kD = bpprix::decode<bpprix::kK>(T) + 2; // Number of spline points
        static constexpr unsigned short P   = bpprix::decode<bpprix::kP>(T); // Number of posterior samples
        static constexpr unsigned short PR = P*bpprix::decode<bpprix::kR>(T);
        static constexpr MKL_UINT PRK       = P*RK;

        template <unsigned short... Ts> struct Factor; 

      /**
        * @brief Randomly selects a birth, death or change action 
        *              depending on what the current number of ridge functions allows. 
        */
        action_t getAction(const unsigned short nRidge) {
            action_t raction = action::kNone;
            const bool expandable = nRidge < bpprix::decode<bpprix::kR>(T);
            _knots.runif(expandable ? action::kBirth : action::kDeath, nRidge > 1 ? action::N : action::kDeath, raction);
            return raction;
        }

      /**
        * @brief Initial Metropolis Hastings terms depending only on the number of ridge functions. Just the Spageiria 
        *              (dealing with life state changes) term currently contributes, taken directly from the Collins et al R code. 
        */
        template<mhix_t U> float getMH(const unsigned short nRidge);
        template<> float getMH<mhix::kSpageiria>(const unsigned short nRidge) {
            return nRidge == 0 ? 0.0f : (nRidge < bpprix::decode<bpprix::kR>(T) ? cx::log(3.0f) : cx::log(2.0f));
        }

      /**
        * @brief Spline basis over all N samples from a knots+projection package for (normally new) ridge rix. 
        * 
        * @note The first basis vector, calculated in the last vsFdimI expression here,  is the projection relu'd 
        *              with the zeroth knot as per BPPR. The natural spline however would just have 
        *              std::memcpy(_basis, projection, N*sizeof(float));
        *              TODO: Consider using the unmodfied natural spline expression instead of the modified BPPR one, 
        *                           and also using a linear (rather than cubic) extension for the K'th basis vector
        */
        void toBasis(const unsigned short rix, const Knots<T>::gettable_t& knojection) {
            const auto& [knotsx, projection] = knojection;
            float* basisr = _basis+N*((kD-2)*rix + bpprix::decode<bpprix::kI>(T));

            vsFdimI(N, projection, SINGLESTEP, knotsx+kD-1, NOSTEP, _working, SINGLESTEP);
            vsPowx(N, _working, 3, _working);
            for (unsigned short ix = kD-2; ix > 0; --ix) {
                float* basisx = basisr+N*(ix-1);
                vsFdimI(N, projection, SINGLESTEP, knotsx+ix, NOSTEP, basisx, SINGLESTEP);
                vsPowx(N, basisx, 3, basisx);
                vsSub(N, basisx, _working, basisx);
                const float kdiff = knotsx[kD-1] - knotsx[ix];
                vsDivI(N, basisx, SINGLESTEP, &kdiff, NOSTEP, basisx, SINGLESTEP);
            }
            std::memcpy(_working, basisr+N*(kD-3), N*sizeof(float));
            for (unsigned short ix = kD-3; ix > 0; --ix) { 
                float* basisx = basisr+N*(ix-1);
                vsSub(N, basisx, _working, basisx + N);
            }
            vsFdimI(N, projection, SINGLESTEP, knotsx, NOSTEP, basisr, SINGLESTEP);
        }   

      /**
        * @brief The fast form of spline basis generation for prediction, where a single-sample regressor X 
        *              gives up to R*P projections
        */
        template<unsigned short U>
        void toBasis(const MKL_UINT(&rcx)[rcx::N], const Knots<T>::template predable_t<U>& knojection, float(&out)[U*RK]) {
            if (rcx[rcx::kRx] > N) { LOGERR("[ZellnerSiow::toBasis] Buffer Overrun %u given N %u", prec::_, rcx[rcx::kRx], N); }
            const MKL_INT K = kD-2;
            const auto& [knots, projection] = knojection;
            // knots here is the entire circular buffer and needs indexing by its head & tail
            const MKL_INT r = rcx[rcx::kKHx] <= rcx[rcx::kKTx] ? rcx[rcx::kKHx] : 0;
            const MKL_INT n = ((r>0 || !rcx[rcx::kKHx]) ?  PR : rcx[rcx::kKHx]) - rcx[rcx::kKTx]; 
            const float* const knotsx = knots+rcx[rcx::kKTx]*kD;
            vsFdimI(n, projection, SINGLESTEP, knotsx+kD-1, kD, _working, SINGLESTEP);
            if (r > 0) { vsFdimI(r, projection+n, SINGLESTEP, knots+kD-1, kD, _working+n, SINGLESTEP); }
            vsPowx(rcx[rcx::kRx], _working, 3, _working);
            for (unsigned short ix = K; ix > 0; --ix) {
                vsFdimI(n, projection, SINGLESTEP, knotsx+ix, kD, out+ix-1, K);
                if (r > 0) { vsFdimI(r, projection+n, SINGLESTEP, knots+ix, kD, out+n*K+ix-1, K); }
            }
            vsPowx(rcx[rcx::kRx]*K, out, 3, out);
            for (unsigned short ix = K; ix > 0; --ix) {
                vsSubI(rcx[rcx::kRx], out+ix-1, K, _working, SINGLESTEP, out+ix-1, K);
            }
            for (unsigned short ix = K; ix > 0; --ix) {
                vsSubI(n, knotsx+kD-1, kD, knotsx+ix, kD, _working, SINGLESTEP);
                if (r > 0) { vsSubI(r, knots+kD-1, kD, knots+ix, kD, _working+n, SINGLESTEP); }
                vsDivI(rcx[rcx::kRx], out+ix-1, K, _working, SINGLESTEP, out+ix-1, K);
            }
            scopy(reinterpret_cast<const MKL_INT*>(&rcx[rcx::kRx]), out+kD-3, &K, _working, &SINGLESTEP);
            for (unsigned short ix = kD-3; ix > 0; --ix) { 
                vsSubI(rcx[rcx::kRx], out+ix-1, K, _working, SINGLESTEP, out+ix, K);
            }
            vsFdimI(n, projection, SINGLESTEP, knotsx, kD, out, K);
            if (r > 0) { vsFdimI(r, projection+n, SINGLESTEP, knots, kD, out, K); }
        }
      /**
        * @brief The specialization of the fast form of spline basis generation for prediction, to the single-sample update case,
        *              where it is in fact the current knots and not cache from knots::get
        */
        void toBasis(const MKL_INT &rcx, const Knots<T>::template predable_t<1>& knojection, float(&out)[RK]) {
            const MKL_INT K = kD-2;
            const auto& [knots, projection] = knojection;

            vsFdimI(rcx, projection, SINGLESTEP, knots+kD-1, kD, _working, SINGLESTEP);
            vsPowx(rcx, _working, 3, _working);
            for (unsigned short ix = K; ix > 0; --ix) {
                vsFdimI(rcx, projection, SINGLESTEP, knots+ix, kD, out+ix-1, K);
            }
            vsPowx(rcx*K, out, 3, out);
            for (unsigned short ix = K; ix > 0; --ix) {
                vsSubI(rcx, out+ix-1, K, _working, SINGLESTEP, out+ix-1, K);
            }
            for (unsigned short ix = K; ix > 0; --ix) {
                vsSubI(rcx, knots+kD-1, kD, knots+ix, kD, _working, SINGLESTEP);
                vsDivI(rcx, out+ix-1, K, _working, SINGLESTEP, out+ix-1, K);
            }
            scopy(&rcx, out+kD-3, &K, _working, &SINGLESTEP);
            for (unsigned short ix = kD-3; ix > 0; --ix) { 
                vsSubI(rcx, out+ix-1, K, _working, SINGLESTEP, out+ix, K);
            }
            vsFdimI(rcx, projection, SINGLESTEP, knots, kD, out, K);
        }

      /**
        * @brief The equivalent for blas call sgemv with TRANSPOSE but for RFPF lower-triangular cholesky matrix
        *              and where the multiplying vector z is overwritten with the result
        */
        template<factor_t F>
        void sgemvtrf(Factor<F>& f, float(&z)[Factor<F>::kM]) {
            const auto& rfpf = f.rfpf();

            strmv(&LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &rfpf.m1, rfpf.lo, &rfpf.ki, z, &SINGLESTEP);                                             // L_{1,1}^T * z_{1}
            sgemv(&TRANSPOSED, &rfpf.m2, &rfpf.m1, &ONEf, rfpf.lo+rfpf.m1, &rfpf.ki, z+rfpf.m1, &SINGLESTEP, &ONEf, z, &SINGLESTEP); // L_{2,1}^T * z_{2} + L_{1,1}^T * z_{1}
            strmv(&UPPERTRIANGLE, &UNTRANSPOSED, &NONUNITARY, &rfpf.m2, rfpf.up, &rfpf.ki, z+rfpf.m1, &SINGLESTEP);                       // L_{2,2}^T * z_{2}
        }

      /**
        * @brief RFPF Rank One Update of the lower-triangular cholesky matrix, a la Krause-Igel
        */
        template<cholesky_t U, factor_t V=factor::kProposal, factor_t F> requires update::rankOne_t<U> 
        void update(Factor<F>& f, float(&b)[Factor<F>::kM], const float alpha, const float beta, const unsigned short subx) {
            const auto& rfpf = f.template rfpf<V>();
            const float isqra = 1 / std::sqrt(alpha);
            float bx = 1;
            for (unsigned short ix = subx; ix < f.template k<V>(); ++ix) {
                float* const ixp = ix < rfpf.m1 ? rfpf.lo+ix*(rfpf.ki+1) : rfpf.up+(ix-rfpf.m1)*(rfpf.ki+1);
                const float lx = *ixp * isqra;
                const float lx2 = alpha*alpha * lx*lx;
                const float gamma = lx2*bx + beta*b[ix]*b[ix];
                *ixp = std::sqrt(gamma / bx);
                const MKL_INT stride = ix < rfpf.m1 ? SINGLESTEP : rfpf.ki;
                float* const lpp = ixp+stride;
                const MKL_INT kx = f.template k<V>()-ix-1;
                float lc = b[ix] / lx;
                if (lc == 0) {
                    lc = *ixp / lx;
                    vsMulI(kx, lpp, stride, &lc, NOSTEP, lpp, stride);
                } else {
                    vsMulI(kx, lpp, stride, &lc, NOSTEP, lpp, stride);
                    vsSubI(kx, b+ix+1, SINGLESTEP, lpp, stride, b+ix+1, SINGLESTEP);
                    lc = *ixp / b[ix];
                    vsMulI(kx, lpp, stride, &lc, NOSTEP, lpp, stride);
                    lc =  *ixp*beta*b[ix] / gamma;
                    saxpy(&kx, &lc, b+ix+1, &SINGLESTEP, lpp, &stride);
                }

                bx = gamma / lx2; 
            }
        }
        template<cholesky_t U, factor_t V=factor::kProposal, factor_t F> requires update::rankOne_t<U> 
        void update(Factor<F>& f, float(&b)[Factor<F>::kM], const float alpha, const float beta) {
            update<U, V>(f, b, alpha, beta, 0); 
        }

      /**
        * @brief RFPF Full Rank Update of the lower-triangular cholesky matrix. 
        * 
        * @details Lapack has 'spftrf', a function that calculates the cholesky factorization for RFPF matrices, but the expected
        *                  format only coincides with the Factor::RFPF format when of even order. Unfortunately the Lapack RFPF functions
        *                  have not been extended to account for different leading dimensions, so in the case of an odd order matrix we 
        *                  have to handle the RFPF sub-operations ourselves.  
        */
        template<cholesky_t U, factor_t F> requires update::fullRank_t<U> 
        void update(Factor<F>& f) {
            const auto& rfpf = f.rfpf();
            ssyrk(&LOWERTRIANGLE, &TRANSPOSED, &rfpf.m1, &f.kN, &ONEf, f.src(), &f.kN, &ZEROf, rfpf.lo, &rfpf.ki);                                                                           // B_{1,1}^T * B_{1,1}
            ssyrk(&UPPERTRIANGLE, &TRANSPOSED, &rfpf.m2, &f.kN, &ONEf, f.src() + rfpf.m1*f.kN, &f.kN, &ZEROf, rfpf.up, &rfpf.ki);                                                     // B_{2,2}^T * B_{2,2}
            sgemm(&TRANSPOSED, &UNTRANSPOSED, &rfpf.m2, &rfpf.m1, &f.kN, &ONEf, f.src() + rfpf.m1*f.kN, &f.kN, f.src(), &f.kN, &ZEROf, rfpf.lo+rfpf.m1, &rfpf.ki);  // B_{2,1}^T * B_{1,2}

            // Run the cholesky lower triangular algorithm 
            if (f.odd()) {
                spotrf(&LOWERTRIANGLE, &rfpf.m1, rfpf.lo, &rfpf.ki, &_outcome);
                strsm(&RIGHTSIDE, &LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &rfpf.m2, &rfpf.m1, &ONEf, rfpf.lo, &rfpf.ki, rfpf.lo+rfpf.m1, &rfpf.ki);
                ssyrk(&UPPERTRIANGLE, &UNTRANSPOSED, &rfpf.m2, &rfpf.m1, &NEGATIVEONEf, rfpf.lo+rfpf.m1, &rfpf.ki, &ONEf, rfpf.up, &rfpf.ki);   
                spotrf(&UPPERTRIANGLE, &rfpf.m2, rfpf.up, &rfpf.ki, &_outcome);
            } else {
                spftrf(&UNTRANSPOSED, &LOWERTRIANGLE, &f.k(), f.get(), &_outcome); 
            }
        }
        template<cholesky_t U, factor_t F> requires update::fullRank_t<U> 
        void update(Factor<F>& f, const float(&y)[Factor<F>::kN], float(&by)[Factor<F>::kM]) {
            update<U>(f);
            // Compute the new basis-transformed response 'y'
            sgemv(&TRANSPOSED, &f.kN, &f.k(), &ONEf, f.src(), &f.kN, y, &SINGLESTEP, &ZEROf, by, &SINGLESTEP);
        }

      /**
        * @brief Computes X = L^{-T}Z, the vector Z covariance-adjusted by (LL^T)^{-1} where L is in lower RFP Format and 
        *              is the _cholesky member of Factor<F>._control. Overwrites Z with solution X.
        */
        template<factor_t F>
        void covarize(Factor<F>& f, float(&z)[Factor<F>::kM]) {
            const auto& rfpf = f.rfpf();

            strtrs(&UPPERTRIANGLE, &UNTRANSPOSED, &NONUNITARY, &rfpf.m2, &SINGLESTEP, rfpf.up, &rfpf.ki, z+rfpf.m1, &f.k(), &_outcome);         // L_{2,2}^T * X_{2} = Z_{2}
            sgemv(&TRANSPOSED, &rfpf.m2, &rfpf.m1, &NEGATIVEONEf, rfpf.lo+rfpf.m1, &rfpf.ki, z+rfpf.m1, &SINGLESTEP, &ONEf, z, &SINGLESTEP);  // L_{2,1}^T * X_{2}
            strtrs(&LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &rfpf.m1, &SINGLESTEP, rfpf.lo, &rfpf.ki, z, &f.k(), &_outcome);                               // L_{1,1}^T * X_{1} = Z_{1} - L_{2,1}^T * X_{2}
        }

      /**
        * @brief Computes X = L^{-1}Z, the solution to LX=Z and related to the covarize function. 
        *              L is in lower RFP Format and is by default the PROPOSAL; Z is overwritten with solution X.
        */
        template<factor_t U=factor::kProposal, unsigned short M, factor_t F>
        void eziravoc(const Factor<F>& f, float(&z)[M]) {
            if (f.template odd<U>()) {
                const auto& rfpf = f.template rfpf<U>();
                strtrs(&LOWERTRIANGLE, &UNTRANSPOSED, &NONUNITARY, &rfpf.m1, &SINGLESTEP, rfpf.lo, &rfpf.ki, z, &rfpf.m1, &_outcome);
                sgemv(&UNTRANSPOSED, &rfpf.m2, &rfpf.m1, &NEGATIVEONEf, rfpf.lo+rfpf.m1, &rfpf.ki, z, &SINGLESTEP, &ONEf, z+rfpf.m1, &SINGLESTEP);
                strtrs(&UPPERTRIANGLE, &TRANSPOSED, &NONUNITARY, &rfpf.m2, &SINGLESTEP, rfpf.up, &rfpf.ki, z+rfpf.m1, &rfpf.m2, &_outcome); 
            } else {
                spftrs(&UNTRANSPOSED, &LOWERTRIANGLE, &_basesky.template k<U>(), &SINGLESTEP, _basesky.template get<U>(), z, &_basesky.template k<U>(), &_outcome);
            }
        }
      /**
        * @brief The version of eziravoc that solves for multiple RHS. 
        * 
        * @note The number is assumed to be the proposal k - control k and accordingly, L is the CONTROL (not the proposal)
        */
        template<unsigned short M, factor_t F>
        void eziravoc(const Factor<F>& f, float(&z)[M], const MKL_INT& ldz) {
        const MKL_INT K = f.template k<factor::kProposal>() - f.k();
        if (f.odd()) {
                const auto& rfpf = f.rfpf();
                strtrs(&LOWERTRIANGLE, &UNTRANSPOSED, &NONUNITARY, &rfpf.m1, &K, rfpf.lo, &rfpf.ki, z, &ldz, &_outcome);
                sgemm(&UNTRANSPOSED, &UNTRANSPOSED, &rfpf.m2, &K, &rfpf.m1, &NEGATIVEONEf, rfpf.lo+rfpf.m1, &rfpf.ki, z, &ldz, &ONEf, z+rfpf.m1, &ldz);
                strtrs(&UPPERTRIANGLE, &TRANSPOSED, &NONUNITARY, &rfpf.m2, &K, rfpf.up, &rfpf.ki, z+rfpf.m1, &ldz, &_outcome); 
            } else {
                spftrs(&UNTRANSPOSED, &LOWERTRIANGLE, &_basesky.k(), &K, _basesky.get(), z, &ldz, &_outcome);
            }
        }

      /**
        * @brief Solve for the regression basis-coefficients where L, the lower-triangular cholesky factorization 
        *              of inverse-covariance B^{T}B, is in lower RFP Format and stored in the member matrix _cholesky 
        *              and the RHS B^{T}Y is in array _by
        */
        template<action_t A>
        float beta() {
            std::memcpy(_beta, _by, _basesky.k()*sizeof(float));
            eziravoc<factor::kControl>(_basesky, _beta);
            sscal(&_basesky.k(), &_tau, _beta, &SINGLESTEP);
            
            // Update the sse
            return sdot(&_basesky.k(), _by, &SINGLESTEP, _beta, &SINGLESTEP);
        }
        template<action_t A> float beta(const MKL_INT& dx) {} 
        template<>
        float beta<action::kBirth>() {
            // The last (kD-2) cols of _basis are assumed to represent a newly added ridge function. Add B_{new}^T._by here
            // Use _working as the stand-in for _beta; if the proposal is accepted will need to copy into _beta 
            const MKL_INT K = kD-2;
            sgemv(&TRANSPOSED, &Factor<factor::kBasis>::kN, &K, &ONEf, _basesky.src()+_basesky.k()*Factor<factor::kBasis>::kN, &Factor<factor::kBasis>::kN, _y, &SINGLESTEP, &ZEROf, _by+_basesky.k(), &SINGLESTEP);
            std::memcpy(_working, _by, _basesky.template k<factor::kProposal>()*sizeof(float));
            eziravoc(_basesky, _working);
            sscal(&_basesky.template k<factor::kProposal>(), &_tau, _working, &SINGLESTEP);
            return sdot(&_basesky.template k<factor::kProposal>(), _by, &SINGLESTEP, _working, &SINGLESTEP);
        }
        template<>
        float beta<action::kDeath>(const MKL_INT& ex) {
            // Copy all-but-_by_{ex} (ie B_{ex}^T._y) into _working, which is in turn the stand-in for _beta;
            // if the proposal is accepted will need to copy into _beta 
            const MKL_INT K = kD-2;
            const MKL_INT pm2 = _basesky.template k<factor::kProposal>()-ex;
            std::memcpy(_working, _by, ex*sizeof(float));
            std::memcpy(_working+ex, _by+ex+K, pm2*sizeof(float));
            eziravoc(_basesky, _working);
            sscal(&_basesky.template k<factor::kProposal>(), &_tau, _working, &SINGLESTEP);
            return sdot(&ex, _by, &SINGLESTEP, _working, &SINGLESTEP) 
                        + sdot(&pm2, _by+ex+K, &SINGLESTEP, _working+ex, &SINGLESTEP);
        }
        template<>
        float beta<action::kChange>(const MKL_INT& rx) {
            // Replace _by_{rx} (ie B_{rx}^T._y) with B_{_ldb}^T._y into _working, which is in turn the stand-in for _beta;
            // if the proposal is accepted will need to copy into _beta 
            const MKL_INT K = kD-2;
            const MKL_INT pm2 = _basesky.k()-rx-K;
            std::memcpy(_working, _by, rx*sizeof(float));
            std::memcpy(_working+rx+K, _by+rx+K, pm2*sizeof(float));
            sgemv(&TRANSPOSED, &Factor<factor::kBasis>::kN, &K, &ONEf, _basesky.src()+_basesky.k()*Factor<factor::kBasis>::kN, &Factor<factor::kBasis>::kN, _y, &SINGLESTEP, &ZEROf, _working+rx, &SINGLESTEP);
            std::memcpy(_working+_basesky.k(), _working+rx, K*sizeof(float));
            eziravoc(_basesky, _working);
            sscal(&_basesky.k(), &_tau, _working, &SINGLESTEP);
            return sdot(&rx, _by, &SINGLESTEP, _working, &SINGLESTEP) 
                        + sdot(&pm2, _by+rx+K, &SINGLESTEP, _working+rx+K, &SINGLESTEP)
                        + sdot(&K, _working+_basesky.k(), &SINGLESTEP, _working+rx, &SINGLESTEP);
        }

      /**
        * @brief Add the regression coefficients to the ring-buffered cache, updating the head and tail 
        *              indices and also caching the number of ridge functions for tracking purposes. 
        */
        void cache(MKL_UINT(&rcx)[rcx::N]) {
            const unsigned short nROut = rcx[rcx::kPn] >= P ? _cache.first[rcx[rcx::kPx]] : 0;
            LOGVER("[ZellnerSiow::cache] Rolling nRidges On: %u Off: %u", prec::_, _projesky.k(), nROut);
            _knots.cache(rcx, _projesky.k(), nROut);
            _cache.first[rcx[rcx::kPx]] = _projesky.k();
            // _cache.second is populated from a beta sample every sample() call. 
            rcx[rcx::kPn] += 1;
            rcx[rcx::kPx] = rcx[rcx::kPn] % P;
            rcx[rcx::kRx] += _projesky.k() - nROut;
            LOGVER("[ZellnerSiow::cache] kPx -> %u kRx -> %u ", prec::_, rcx[rcx::kPx], rcx[rcx::kRx]);
            rcx[rcx::kRKx] += (bpprix::decode<bpprix::kI>(T) ? _basesky.k()-1 : _basesky.k()) - nROut*bpprix::decode<bpprix::kK>(T);
            rcx[rcx::kZHx] += (bpprix::decode<bpprix::kI>(T) ? _basesky.k()-1 : _basesky.k());
            if (rcx[rcx::kZHx] >= PRK) { rcx[rcx::kZHx] -= (PRK - bpprix::decode<bpprix::kI>(T) ? P : 0); }
            if (rcx[rcx::kPn] >= P) {
                rcx[rcx::kZTx] += nROut*bpprix::decode<bpprix::kK>(T);
                if (rcx[rcx::kZTx] >= PRK) { rcx[rcx::kZTx] -= (PRK - bpprix::decode<bpprix::kI>(T) ? P : 0); }
            }
            LOGVER("[ZellnerSiow::cache] kZTx -> %u kZHx -> %u ", prec::_, rcx[rcx::kZTx], rcx[rcx::kZHx]);
        }

      /**
        * @brief Draw Bayesian samples for the Zellner Siow quantities (beta and its hyperparameters sigma & tau)
        */
        void sample(MKL_UINT(&rcx)[rcx::N]) { 
            _knots.rigamma(N/2, 2/_sse, _sigma);
            _sigma = std::sqrt(_sigma);

            _knots.rsnormal(_basesky.k(), _working);
            covarize(_basesky, reinterpret_cast<float(&)[Factor<factor::kBasis>::kM]>(_working));
            _sse = _tau;                                   // Caching about-to-be-previous tau value
            _tau = std::sqrt(_tau) * _sigma; // Reuse of scalar tau that will be set below anyway
            MKL_INT n = std::min(bpprix::decode<bpprix::kI>(T) ? _basesky.k()-1 : _basesky.k(), static_cast<MKL_INT>(PRK-rcx[rcx::kZHx]));
            float* coef = _cache.second+rcx[rcx::kZHx];
            float* const z = bpprix::decode<bpprix::kI>(T) ? _working+1 : _working;
            std::memcpy(coef, bpprix::decode<bpprix::kI>(T) ? _beta+1 : _beta, n*sizeof(float));
            saxpy(&n, &_tau, z, &SINGLESTEP, coef, &SINGLESTEP);
            std::memcpy(z, coef , n*sizeof(float));
            if constexpr(bpprix::decode<bpprix::kI>(T)) {
                _cache.second+rcx[rcx::kPx] = _working[0] = _beta[0] + _tau*_working[0];
            }
            // Wrap around circular buffer _cache.second if necessary
            n = (bpprix::decode<bpprix::kI>(T) ? _basesky.k()-1 : _basesky.k()) - n;
            if (n > 0) {
                const unsigned short ni = _basesky.k()-n;
                coef = _cache.second + (bpprix::decode<bpprix::kI>(T) ? P : 0);
                std::memcpy(coef, _beta+ni, n*sizeof(float));
                saxpy(&n, &_tau, _working+ni, &SINGLESTEP, coef, &SINGLESTEP);
                std::memcpy(_working+ni, coef, n*sizeof(float));
            }
            sgemvtrf(_basesky, reinterpret_cast<float(&)[Factor<factor::kBasis>::kM]>(_working));
            _tau = sdot(&_basesky.k(), _working, &SINGLESTEP, _working, &SINGLESTEP);

            _knots.rigamma(_dscr.kBetaIGVShape + _basesky.k()/2, 1/(_dscr.kBetaIGVScale + _tau/(2*_sigma*_sigma)), _tau);
            _tau = _tau / (_tau + 1);
            _sxy *= _tau / _sse;
            _sse = std::max(_ssy - _sxy, 1.0f);  // Possible for _sxy > _ssy when performing online lazy-updating. 
        }

      /**
        * @brief Update the proposal cholesky matrix with a modified ridge function which is assumed to be 
        *              already calculated and adjacent to the existing source (_basis/_transform)
        */
        template<unsigned short K, cholesky_t U, factor_t F> requires update::fullRank_t<U> 
        void modifyrfpf(Factor<F>& f, const MKL_INT r) {
            f.template set<factor::kProposal>(f.k());
            const auto& rfpfi = f.rfpf();
            const auto& rfpfo = f.template rfpf<factor::kProposal>();

            const MKL_INT kd = K;
            const bool inupper = r <= rfpfi.m1;
            const MKL_INT kc = f.k() - r - K;
            const MKL_INT rc = inupper ? kc - rfpfi.m1 : r - rfpfi.m1;

            if (r > 0) {
                const MKL_INT nix = inupper ? r : rfpfi.m1;
                // Copy L_{1,1}, L_{3,1} from the control
                for (unsigned short ix = 0; ix < nix; ++ix) { 
                    const unsigned short nex = ix*(rfpfi.ki+1);
                    const unsigned short nfx = nex+r+K-ix;
                    std::memcpy(rfpfo.lo+nex, rfpfi.lo+nex, (r-ix)*sizeof(float));
                    std::memcpy(rfpfo.lo+nfx, rfpfi.lo+nfx, kc*sizeof(float));
                }
                // Place the B_{2,1} submatrix
                if constexpr(F & factor::kBasis) {
                    sgemm(&TRANSPOSED, &UNTRANSPOSED, &kd, &nix, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, f.src(), &Factor<F>::kN, &ZEROf, rfpfo.lo+r, &rfpfo.ki);
                } else if constexpr(F & factor::kProj) { 
                    _knots.cos(0, nix, f.k(), rfpfo.lo+r, rfpfo.ki, ZEROf);
                }
                // Solve for the first min(r, m1) cols of L_{2,1}
                strsm(&RIGHTSIDE, &LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &kd, &nix, &ONEf, rfpfo.lo, &rfpfo.ki, rfpfo.lo+r, &rfpfo.ki);
                // Preliminary L_{2,1}^T . L_{2,1} calculations for solving for L_{2,2}
                if (r < rfpfi.m1) {
                    const MKL_INT kw = std::min(rfpfi.m1-r, kd);
                    float* const of = rfpfo.lo+r*(rfpfo.ki+1);
                    ssyrk(&LOWERTRIANGLE, &UNTRANSPOSED, &kw, &nix, &NEGATIVEONEf, rfpfo.lo+r, &rfpfo.ki, &ZEROf, of, &rfpfo.ki);
                    const MKL_INT kv = K-kw;
                    if (kv > 0) {
                        sgemm(&UNTRANSPOSED, &TRANSPOSED, &kv, &kw, &nix, &NEGATIVEONEf, rfpfo.lo+r+kw, &rfpfo.ki, rfpfo.lo+r, &rfpfo.ki, &ZEROf, of+kw, &rfpfo.ki); 
                    }
                }
                if (r+K > rfpfi.m1) {
                    const MKL_INT kw = std::min(r+K-rfpfi.m1, kd);
                    ssyrk(&UPPERTRIANGLE, &UNTRANSPOSED, &kw, &nix, &NEGATIVEONEf, rfpfo.lo+r+(K-kw), &rfpfo.ki, &ZEROf, rfpfo.up+std::max(r-rfpfo.m1, 0)*(rfpfo.ki+1), &rfpfo.ki);
                }
                // Solve for the complementary cols if necessary
                if (!inupper) {
                    // Copy in the wrapped triangular part of L_{1,1}
                    for (unsigned short ix = 0; ix < rc; ++ix) { 
                        std::memcpy(rfpfo.up+rfpfo.ki*ix, rfpfi.up+rfpfi.ki*ix, (ix+1)*sizeof(float));
                    }
                    float* const of = rfpfo.up+rfpfo.ki*rc;
                    if constexpr(F & factor::kBasis) {
                        sgemm(&TRANSPOSED, &UNTRANSPOSED, &rc, &kd, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*nix, &Factor<F>::kN, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, &ZEROf, of, &rfpfo.ki);
                    } else if constexpr(F & factor::kProj) { 
                        _knots.cos(nix, rc, f.k(), reinterpret_cast<float(&)[bpprix::decode<bpprix::kR>(T)]>(*of));
                    }
                    sgemm(&UNTRANSPOSED, &TRANSPOSED, &rc, &kd, &nix, &NEGATIVEONEf, rfpfi.lo+nix, &rfpfi.ki, rfpfo.lo+r, &rfpfo.ki, &ONEf, of, &rfpfo.ki);
                    strtrs(&UPPERTRIANGLE, &TRANSPOSED, &NONUNITARY, &rc, &kd, rfpfi.up, &rfpfi.ki, of, &rfpfo.ki, &_outcome);  
                    // Further preliminary L_{2,1}^T . L_{2,1} calculations for solving for L_{2,2}
                    ssyrk(&UPPERTRIANGLE, &TRANSPOSED, &kd, &rc, &NEGATIVEONEf, of, &rfpfo.ki, &ONEf, rfpfo.up+rc*(rfpfo.ki+1), &rfpfo.ki);
                }
                if (kc > 0) {
                    // Preliminary L_{3,1}.L_{2,1}^T calculations for solving for L_{3,2}
                    if (r < rfpfi.m1) {
                        const MKL_INT kw = std::min(rfpfi.m1-r, kd);
                        sgemm(&UNTRANSPOSED, &TRANSPOSED, &kc, &kw, &nix, &NEGATIVEONEf, rfpfo.lo+r+K, &rfpfo.ki, rfpfo.lo+r, &rfpfo.ki, &ZEROf, rfpfo.lo+rfpfo.ki*r+r+K, &rfpfo.ki);
                    }
                    if (r+K > rfpfi.m1) {
                        const MKL_INT kw = std::min(r+K-rfpfi.m1, kd);
                        sgemm(&UNTRANSPOSED, &TRANSPOSED, &kw, &kc, &nix, &NEGATIVEONEf, rfpfo.lo+r+K-kw, &rfpfo.ki, rfpfo.lo+r+K, &rfpfo.ki, &ZEROf, rfpfo.up+rfpfo.ki*(r+K-rfpfi.m1)+std::max(r-rfpfi.m1, 0), &rfpfo.ki);
                    }
                    if (!inupper) {
                        // Copy over the wrapped part of L_{3,1}
                        for (unsigned short ix = 0; ix < kc; ++ix) { 
                            const unsigned short nex = (rfpfi.m2-kc+ix)*rfpfi.ki;
                            std::memcpy(rfpfo.up+nex, rfpfi.up+nex, rc*sizeof(float));
                        }
                        sgemm(&TRANSPOSED, &UNTRANSPOSED, &kd, &kc, &rc, &NEGATIVEONEf, rfpfo.up+rfpfo.ki*rc, &rfpfo.ki, rfpfo.up+rfpfo.ki*(rc+K), &rfpfo.ki, &ONEf, rfpfo.up+rfpfo.ki*(rc+K)+rc, &rfpfo.ki);
                    }
                }
            }
            // Solve for L_{2,2}
            if (r < rfpfi.m1) {
                const MKL_INT kw = std::min(rfpfi.m1-r, kd);
                float* const of = rfpfo.lo+r*(rfpfo.ki+1);
                if constexpr(F & factor::kBasis) {
                    ssyrk(&LOWERTRIANGLE, &TRANSPOSED, &kw, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, r > 0 ? &ONEf : &ZEROf, of, &rfpfo.ki);
                } else if constexpr(F & factor::kProj) { 
                    _knots.cos(f.k(), 1, f.k(), of, SINGLESTEP, r > 0 ? ONEf : ZEROf);
                }
                spotrf(&LOWERTRIANGLE, &kw, of, &rfpfo.ki, &_outcome);
                const MKL_INT kv = K-kw;
                if (kv > 0) {
                    sgemm(&TRANSPOSED, &UNTRANSPOSED, &kv, &kw, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*(f.k()+kw), &Factor<F>::kN, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, r > 0 ? &ONEf : &ZEROf, of+kw, &rfpfo.ki); 
                    strsm(&RIGHTSIDE, &LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &kv, &kw, &ONEf, of, &rfpfo.ki, of+kw, &rfpfo.ki);
                    ssyrk(&UPPERTRIANGLE, &UNTRANSPOSED, &kv, &kw, &NEGATIVEONEf, of+kw, &rfpfo.ki, r > 0 ? &ONEf : &ZEROf, rfpfo.up, &rfpfo.ki);
                }
            }
            if (r+K > rfpfi.m1) {
                const MKL_INT kw = std::min(r+K-rfpfi.m1, kd);
                float* const of = rfpfo.up+std::max(r-rfpfo.m1, 0)*(rfpfo.ki+1);
                if constexpr(F & factor::kBasis) {
                    ssyrk(&UPPERTRIANGLE, &TRANSPOSED, &kw, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*(f.k()+K-kw), &Factor<F>::kN, ((r>0) || (kw<K)) ? &ONEf : &ZEROf, of, &rfpfo.ki);
                } else if constexpr(F & factor::kProj) { 
                    _knots.cos(f.k(), 1, f.k(), of, SINGLESTEP, r > 0 ? ONEf : ZEROf);
                }
                spotrf(&UPPERTRIANGLE, &kw, of, &rfpfo.ki, &_outcome);
            }
            if (kc > 0) {
                // Add the B_{3,2} submatrix in the locations used by L{3,1}.L_{2,1} ^T above & solve for L_{3,2}
                if (r < rfpfi.m1) {
                    const MKL_INT kw = std::min(rfpfi.m1-r, kd);
                    if constexpr(F & factor::kBasis) {
                        sgemm(&TRANSPOSED, &UNTRANSPOSED, &kc, &kw, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*(r+K), &Factor<F>::kN, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, r > 0 ? &ONEf : &ZEROf, rfpfo.lo+rfpfo.ki*r+r+K, &rfpfo.ki);
                    } else if constexpr(F & factor::kProj) { 
                        _knots.cos(r+K, kc, f.k(), rfpfo.lo+rfpfo.ki*r+r+K, SINGLESTEP, r > 0 ? ONEf : ZEROf);
                    }
                    float* of = rfpfo.lo+r*(rfpfo.ki+1);
                    strsm(&RIGHTSIDE, &LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &kc, &kw, &ONEf, of, &rfpfo.ki, of+K, &rfpfo.ki);
                    if (kw < K) {
                        sgemm(&UNTRANSPOSED, &TRANSPOSED, &kw, &kc, &kw, &NEGATIVEONEf, of+kw, &rfpfo.ki, of+K, &rfpfo.ki, &ONEf, rfpfo.up+rfpfo.ki*(r+K-rfpfo.m1), &rfpfo.ki); 
                    }
                }
                if (r+K > rfpfi.m1) {
                    const MKL_INT kw = std::min(r+K-rfpfi.m1, kd);
                    float* const of = rfpfo.up+rfpfo.ki*(r+K-rfpfo.m1)+std::max(r-rfpfo.m1, 0);
                    if constexpr(F & factor::kBasis) {
                        sgemm(&TRANSPOSED, &UNTRANSPOSED, &kw, &kc, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*(f.k()+K-kw), &Factor<F>::kN, f.src()+Factor<F>::kN*(r+K), &Factor<F>::kN, r > 0 ? &ONEf : &ZEROf, of, &rfpfo.ki);
                    } else if constexpr(F & factor::kProj) { 
                        _knots.cos(r+K, kc, f.k(), of, rfpfo.ki, r > 0 ? ONEf : ZEROf);
                    }
                    strtrs(&UPPERTRIANGLE, &TRANSPOSED, &NONUNITARY, &kw, &kc, rfpfo.up+std::max(r-rfpfo.m1, 0)*(rfpfo.ki+1), &rfpfo.ki, of, &rfpfo.ki, &_outcome);
                }
                // Copy over L_{3,3}
                const unsigned short mix = std::max(r+K-rfpfi.m1, 0);
                if (inupper) {
                    for (unsigned short ix = r+K; ix < rfpfi.m1; ++ix) { 
                        const unsigned short nex = ix*(rfpfi.ki+1);
                        std::memcpy(rfpfo.lo+nex, rfpfi.lo+nex, (f.k()-ix)*sizeof(float));
                    }
                    for (unsigned short ix = 0; ix < rfpfi.m2-mix; ++ix) { 
                        const unsigned short nex = mix*(rfpfi.ki+1)+rfpfi.ki*ix;
                        std::memcpy(rfpfo.up+nex, rfpfi.up+nex, (ix+1)*sizeof(float));
                    }
                } else {
                    for (unsigned short ix = 0; ix < kc; ++ix) { 
                        const unsigned short nex = (rfpfi.m2-kc+ix)*rfpfi.ki+mix;
                        std::memcpy(rfpfo.up+nex, rfpfi.up+nex, (ix+1)*sizeof(float));
                    }
                }
                // Perform the 2K rank-1 updates to L_{3,3}
                for (unsigned short ix = 0; ix < K; ++ix) {  
                    if (r+ix < rfpfi.m1) {
                        std::memcpy(_working+r+K, rfpfi.lo+(r+ix)*rfpfi.ki+r+K, kc*sizeof(float));
                    } else {
                        scopy(&kc, rfpfi.up+(rfpfi.m2-kc)*rfpfi.ki+r+ix-rfpfi.m1, &rfpfi.ki, _working+r+K, &SINGLESTEP);
                    }
                    update<update::kRankOne>(f, reinterpret_cast<float(&)[Factor<F>::kM]>(_working), ONEf, ONEf, r+K);
                }
                for (unsigned short ix = 0; ix < K; ++ix) {  
                    if (r+ix < rfpfo.m1) {
                        std::memcpy(_working+r+K, rfpfo.lo+(r+ix)*rfpfo.ki+r+K, kc*sizeof(float));
                    } else {
                        scopy(&kc, rfpfo.up+(rfpfo.m2-kc)*rfpfo.ki+r+ix-rfpfo.m1, &rfpfo.ki, _working+r+K, &SINGLESTEP);
                    }
                    update<update::kRankOne>(f, reinterpret_cast<float(&)[Factor<F>::kM]>(_working), ONEf, NEGATIVEONEf, r+K);
                }
            }
        }

      /**
        * @brief Update the proposal cholesky matrix with a new additional ridge function which is assumed to be 
        *              already calculated and adjacent to the existing source (_basis/_transform)
        */
        template<unsigned short K, cholesky_t U, factor_t F> requires update::fullRank_t<U> 
        void expandrfpf(Factor<F>& f) {
            f.template set<factor::kProposal>(f.k()+K);
            const MKL_INT kd = K;
            const auto& rfpfi = f.rfpf();
            const auto& rfpfo = f.template rfpf<factor::kProposal>();

            // Copy/Constuct L_{1,1} in the proposal
            for (unsigned short ix = 0; ix < rfpfi.m1; ++ix) {
                std::memcpy(rfpfo.lo+ix*(rfpfo.ki+1), rfpfi.lo+ix*(rfpfi.ki+1), (f.k()-ix)*sizeof(float));
            }
            const unsigned short md = rfpfo.m1-rfpfi.m1;
            for (unsigned short ix = 0; ix < md; ++ix) {
                const MKL_INT n2 = rfpfi.m2-ix;
                scopy(&n2, rfpfi.up+ix*(rfpfi.ki+1), &rfpfi.ki, rfpfo.lo+(rfpfi.m1+ix)*(rfpfo.ki+1), &SINGLESTEP);
            }
            for (unsigned short ix = 0; ix < rfpfi.m2-md; ++ix) {
                std::memcpy(rfpfo.up+ix*rfpfo.ki, rfpfi.up+(md+ix)*rfpfi.ki+md, (ix+1)*sizeof(float));
            }

            // Solve for the first m1o cols of L_{2,1}
            if constexpr(F & factor::kBasis) {
                sgemm(&TRANSPOSED, &UNTRANSPOSED, &kd, &rfpfo.m1, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, f.src(), &Factor<F>::kN, &ZEROf, rfpfo.lo+f.k(), &rfpfo.ki);
            } else if constexpr(F & factor::kProj) { 
                _knots.cos(0, rfpfo.m1, f.k(), rfpfo.lo+f.k(), rfpfo.ki, ZEROf);
            }
            strsm(&RIGHTSIDE, &LOWERTRIANGLE, &TRANSPOSED, &NONUNITARY, &kd, &rfpfo.m1, &ONEf, rfpfo.lo, &rfpfo.ki, rfpfo.lo+f.k(), &rfpfo.ki);

            // Solve for the next m2o cols of L_{2,1}, which are actually placed in the upper right
            const MKL_INT n2 = f.k()-rfpfo.m1;
            float* const upr = rfpfo.up+rfpfo.ki*(rfpfo.m2-kd);
            if constexpr(F & factor::kBasis) {
                sgemm(&TRANSPOSED, &UNTRANSPOSED, &n2, &kd, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*rfpfo.m1, &Factor<F>::kN, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, &ZEROf, upr, &rfpfo.ki);
                sgemm(&UNTRANSPOSED, &TRANSPOSED, &n2, &kd, &rfpfo.m1, &NEGATIVEONEf, rfpfo.lo+rfpfo.m1, &rfpfo.ki, rfpfo.lo+f.k(), &rfpfo.ki, &ONEf, upr, &rfpfo.ki);
            } else if constexpr(F & factor::kProj) { 
                _knots.cos(rfpfo.m1, n2, f.k(), reinterpret_cast<float(&)[bpprix::decode<bpprix::kR>(T)]>(*upr));
            }
            strsm(&LEFTSIDE, &UPPERTRIANGLE, &TRANSPOSED, &NONUNITARY, &n2, &kd, &ONEf, rfpfo.up, &rfpfo.ki, upr, &rfpfo.ki);

            // Solve for L_{2,2}
            float* const upd = upr + (rfpfo.m2-kd);
            ssyrk(&UPPERTRIANGLE, &UNTRANSPOSED, &kd, &rfpfo.m1, &NEGATIVEONEf, rfpfo.lo+f.k(), &rfpfo.ki, &ZEROf, upd, &rfpfo.ki);
            ssyrk(&UPPERTRIANGLE, &TRANSPOSED, &kd, &n2, &NEGATIVEONEf, upr, &rfpfo.ki, &ONEf, upd, &rfpfo.ki);
            if constexpr(F & factor::kBasis) {
                ssyrk(&UPPERTRIANGLE, &TRANSPOSED, &kd, &Factor<F>::kN, &ONEf, f.src()+Factor<F>::kN*f.k(), &Factor<F>::kN, &ONEf, upd, &rfpfo.ki);
            } else if constexpr(F & factor::kProj) { 
                _knots.cos(f.k(), SINGLESTEP, f.k(), upd, rfpfo.ki, ONEf); // The candidate theta^T.theta 
            }
            spotrf(&UPPERTRIANGLE, &kd, upd, &rfpfo.ki, &_outcome);
        }

      /**
        * @brief Update the proposal cholesky matrix with a removed ridge function.
        */
        template<unsigned short K, cholesky_t U, factor_t F> requires update::fullRank_t<U> 
        void contractrfpf(Factor<F>& f, const unsigned short ex) {
            f.template set<factor::kProposal>(f.k()-K);
            const auto& rfpfi = f.rfpf();
            const auto& rfpfo = f.template rfpf<factor::kProposal>();
            const unsigned short md = rfpfi.m1-rfpfo.m1;
            const unsigned short mu = K-md;

            // Copy over the lower triangular section, skipping the ex rows. 
            MKL_INT nix = ex < rfpfi.m1 ? ex : rfpfi.m1;
            for (unsigned short ix = 0; ix < nix; ++ix) {
                std::memcpy(rfpfo.lo+ix*(rfpfo.ki+1), rfpfi.lo+ix*(rfpfi.ki+1), (ex-ix)*sizeof(float));
                std::memcpy(rfpfo.lo+ix*rfpfo.ki+ex,  rfpfi.lo+ix*rfpfi.ki+ex+K, (f.k()-ex-K)*sizeof(float));
            }
            nix = rfpfi.m1-ex-K;
            if (nix > 0) {
                // Copy the lower-right quadrilateral
                for (unsigned short ix = 0; ix < nix; ++ix) {
                    std::memcpy(rfpfo.lo+(rfpfo.ki+1)*(ix+ex), rfpfi.lo+(rfpfi.ki+1)*(ix+ex+K), (f.k()-K-ex-ix)*sizeof(float));
                }
            }
            nix = rfpfi.m1-ex-mu;
            if (nix > 0) {
                // Copy the transposed top-right RFPF position to its new untransposed lower-right column
                const unsigned short nex = std::min(static_cast<unsigned short>(nix), mu);
                for (unsigned short ix = mu-nex; ix < mu; ++ix) {
                    nix = rfpfi.m2- ix;
                    scopy(&nix, rfpfi.up+(rfpfi.ki+1)*ix, &rfpfi.ki, rfpfo.lo+(rfpfo.ki+1)*(rfpfi.m1-K+ix), &SINGLESTEP);
                }
            }
            if (ex <= rfpfi.m1) {
                // 'Shift' up the upper right triangle 
                nix = ex+mu-rfpfi.m1;
                const MKL_INT nex = std::max(nix, 0);
                nix = rfpfi.m2-nex-mu;
                for (unsigned short ix = 0; ix < nix; ++ix) {
                    std::memcpy(rfpfo.up+rfpfo.ki*(ix+nex)+nex, rfpfi.up+rfpfi.ki*(ix+nex+mu)+nex+mu, (ix+1)*sizeof(float)); 
                }
            }
            nix = ex-rfpfo.m1;
            if (nix > 0) {
                // Copy the last <=md cols to their transposed top-right RFPF position
                const unsigned short nex = std::min(static_cast<unsigned short>(nix), md);
                for (unsigned short ix = 0; ix < nex; ++ix) {
                    nix = rfpfi.m2 - mu - ix;
                    scopy(&nix, rfpfo.lo+(rfpfo.m1+ix)*(rfpfo.ki+1), &SINGLESTEP, rfpfo.up+(rfpfo.ki+1)*ix, &rfpfo.ki);
                }
            }
            nix = ex-rfpfi.m1;
            if (nix > 0) {
                // Copy the upper-left pre-ex triangle 
                for (unsigned short ix = 0; ix < nix; ++ix) {
                    std::memcpy(rfpfo.up+rfpfo.ki*(ix+md)+md, rfpfi.up+rfpfi.ki*ix, (ix+1)*sizeof(float));
                }
                // Copy the upper right post-ex rectangle and the lower-right triangle beneath it
                const unsigned short nex = f.k()-K-ex;
                for (unsigned short ix = 0; ix < nex; ++ix) {
                    const float* const ipx = rfpfi.up+rfpfi.ki*(ix+nix+K);
                    float* const opx = rfpfo.up+rfpfo.ki*(ex-rfpfo.m1+ix)+md;
                    std::memcpy(opx, ipx, nix*sizeof(float));
                    std::memcpy(opx+nix, ipx+nix+K, (ix+1)*sizeof(float));
                }
            }

            // Make the K rank-1 updates to modify the lower-right cholesky block L_{3,3}
            nix = f.k()-K-ex;
            if (nix > 0) {
                for (unsigned short ix = 0; ix < K; ++ix) {
                    const unsigned short eix =  ex+ix;
                    if (eix < rfpfi.m1) {
                        std::memcpy(_working+ex, rfpfi.lo+rfpfi.ki*eix+ex+K, nix*sizeof(float));
                    } else {
                        scopy(&nix, rfpfi.up+rfpfi.ki*(ex-rfpfi.m1+K)+eix-rfpfi.m1, &rfpfi.ki, _working+ex, &SINGLESTEP);
                    }
                    update<update::kRankOne>(f, reinterpret_cast<float(&)[Factor<F>::kM]>(_working), ONEf, ONEf, ex);
                }
            }
        }

      /**
        * @brief The Wallenius term for signal index selection, based on the current adaptive signal weightings and 
        *              accounting for the cumulative structure of the weights in _indexW. Only calculated here for 1, 2 or 3 dimensions. 
        * 
        * @note The index of the selected signal, fix, is effectively one-based here as it indexes into _indexW which 
        *              has first element zero. 
        */
        float dwallenius(const unsigned short(&fix)[1]) {
            return (_indexW[fix[0]]-_indexW[fix[0]-1]) / _indexW[bpprix::decode<bpprix::kM>(T)];
        }
        float dwallenius(const unsigned short(&fix)[2]) {
            const float y = _indexW[bpprix::decode<bpprix::kM>(T)];
            const float x = y - (_indexW[fix[0]] - _indexW[fix[0]-1] + _indexW[fix[1]] - _indexW[fix[1]-1]);
            return 1 + x/y - x/(y-_indexW[fix[0]]+_indexW[fix[0]-1]) - x/(y-_indexW[fix[1]]+_indexW[fix[1]-1]);
        }
        float dwallenius(const unsigned short(&fix)[3]) {
            const float y = _indexW[bpprix::decode<bpprix::kM>(T)];
            const float xi[3] = {_indexW[fix[0]] - _indexW[fix[0]-1], _indexW[fix[1]] - _indexW[fix[1]-1], _indexW[fix[2]] - _indexW[fix[2]-1]};
            const float x = y - xi[0] - xi[1] - xi[2];
            return 1 + x/y - x/(x+xi[0]) - x/(x+xi[1]) - x/(x+xi[2]) + x/(y-xi[0]) + x/(y-xi[1]) + x/(y-xi[2]);
        }
        float dwallenius(const unsigned short nActive) {
            switch (nActive) {
            case 1:
                return dwallenius(reinterpret_cast<unsigned short(&)[1]>(_pmix));
            case 2:
                return dwallenius(reinterpret_cast<unsigned short(&)[2]>(_pmix));
            case 3:
                return dwallenius(reinterpret_cast<unsigned short(&)[3]>(_pmix));
            }
            return 0;
        }

      /**
        * @brief Generate the birth, death or change proposals. Compute the Bayes ratio and accept or reject accordingly. 
        * 
        * @details Shortcut the rejection if the Anti-Similarity measure is -Inf or if a valid set of signal indices cannot be found, given 
        *                 the candidates in the set already chosen and the existing ridge functions. 
        */
        template<action_t A> void propose() {}

        template<>
        void propose<action::kBirth>() {
            float ractive;
            _knots.runif(0.0f, _activeW[bpprix::decode<bpprix::kA>(T)], ractive);
            const unsigned short nActive = std::distance(std::begin(_activeW), std::upper_bound(std::begin(_activeW), std::end(_activeW), ractive));
            // Nott, Kuk & Duc for the number of active signals:
            _mh[mhix::kSelection] =  -cx::log(static_cast<float>(bpprix::decode<bpprix::kA>(T))) - std::log(_activeW[nActive] - _activeW[nActive-1]);

            // TODO: Consider possibility of making nu updateable / Bayesian
            // For now just pull the randomly-initialized or warmstarted one already there.

            unsigned short assignx = 0;
            if (nActive == 1) {
                // Take a signal with uniform probability but without replacement
                MKL_UINT assigned = (1 << bpprix::decode<bpprix::kM>(T)) - 1;
                while (assignx < 1 && assigned > 0) {
                    _knots.runif(assignx, static_cast<unsigned short>(bpprix::decode<bpprix::kM>(T)), _pmix[0]);
                    assignx += !(_knots.has(1 << _pmix[0], _projesky.k()) && (assigned &= ~(1 << _pmix[0])));
                }
                if (assigned == 0) {
                    LOGLAC("[ZellnerSiow::propose] All signals have already been SINGLY selected. Abort this birth proposal.", prec::_, L3_ARG_UNUSED, L3_ARG_UNUSED);
                    return;
                }
                _pmu[_pmix[0]] = 1;
                toBasis(_projesky.k(), _knots.get(_projesky.k(), _pmix[0], _knots.rbool())); 
            } else {
                unsigned short nacx = bpprix::decode<bpprix::kM>(T) - nActive;
                MKL_UINT assigned = 0;
                while (assignx < nActive && nacx > 0) {
                    _knots.runif(0.0f, _indexW[bpprix::decode<bpprix::kM>(T)], ractive);
                    _pmix[assignx] = std::distance(std::begin(_indexW), std::upper_bound(std::begin(_indexW), std::end(_indexW), ractive));
                    assignx += !((assigned >> (_pmix[assignx]-1)) & 1) && (assigned |= (1 << (_pmix[assignx]-1))) 
                                        && (assignx < nActive-1 || !(_knots.has(assigned, _projesky.k(), nActive) && nacx--)); 
                }
                if (nacx == 0) {
                    LOGLAC("[ZellnerSiow::propose] Found only %u of %u candidate signal indices. Abort this birth proposal.", prec::_, assignx, nActive);
                    return;
                }
                // Nott, Kuk & Duc for the signals chosen:
                _mh[mhix::kSelection] -= (lnCr(bpprix::decode<bpprix::kM>(T), nActive) + std::log(dwallenius(nActive)));
                // Return the proposed signal indices to zero-base after one-base-oriented dwallenius
                for (unsigned short ix = 0; ix < nActive; ++ix) { _pmix[ix] -= 1; }

                // Sample from the power spherical distribution
                std::memcpy(_pmu, SPHERICALMU[nActive-1], nActive*sizeof(float));
                _knots.rps(_pmu, 0); // Diffuse prior
                toBasis(_projesky.k(), _knots.get(_projesky.k(), _pmix, _pmu));
            }
            // Projection's _transform has already been updated in the k'th slot with the new direction
            expandrfpf<1, update::kFull>(_projesky);
            _mh[mhix::kPrior] = 2*std::log(_projesky.template dset<factor::kProposal>() / _projesky.det());
            LOGVER("[ZellnerSiow::propose] Birth Proposal Anti-Similarity Ratio %n.%u", prec::_XX, _mh[mhix::kPrior]);
            if (std::isinf(_mh[mhix::kPrior]) || std::isnan(_mh[mhix::kPrior])) {
                LOGLAC("[ZellnerSiow::propose] High Proposal Similarity %n.%u. Abort this birth proposal.", prec::_X, _mh[mhix::kPrior]);
                return;
            }

            expandrfpf<kD-2, update::kFull>(_basesky);
            float pse = beta<action::kBirth>();
            if (pse < _tau*_ssy) {
                _mh[mhix::kProposal] = getMH<mhix::kSpageiria>(_projesky.k()+1);  
                pse = _ssy - pse;
                _mh[mhix::kTotal] = _mh[mhix::kSpageiria] - _mh[mhix::kProposal] + _mh[mhix::kSelection] + _mh[mhix::kPrior]
                                                    - N*(std::log(pse) - std::log(_sse))/2 + std::log(_dscr.kLambda/(_projesky.k()+1)) - (kD-2)*std::log(1+_tau)/2;
                float mhr = 0;
                _knots.runif(0.0f, 1.0f, mhr);
                mhr = std::log(mhr);
                LOGVER("[ZellnerSiow::propose] Birth Proposal SSE %u.%u", prec::_X, pse); APPEND(VERBOSE, "vs Control SSE %u.%u", prec::_X, _sse);
                if (mhr < _mh[mhix::kTotal]) { // Then the proposal is accepted
                    LOGLAC("[ZellnerSiow::propose] Birth ACCEPTED with %n.%u", prec::_XX, mhr); APPEND(LACONIC, "< %n.%u", prec::_XX, _mh[mhix::kTotal]);
                    _projesky.accept();
                    _basesky.accept();
                    _sxy = _ssy - pse;
                    _sse = pse;
                    _mh[mhix::kSpageiria] = _mh[mhix::kProposal];
                    std::memcpy(_beta, _working, _basesky.k()*sizeof(float));

                    // Adapt the nActive & signal weights
                    vsMulI(bpprix::decode<bpprix::kA>(T), _activeW+1, SINGLESTEP, _activeW+bpprix::decode<bpprix::kA>(T)+1, NOSTEP, _activeW+1, SINGLESTEP);
                    vsAddI(bpprix::decode<bpprix::kA>(T)-nActive+2,  _activeW+nActive, SINGLESTEP, &ONEf, NOSTEP, _activeW+nActive, SINGLESTEP);
                    float z = 1 / _activeW[bpprix::decode<bpprix::kA>(T)+1];
                   vsMulI(bpprix::decode<bpprix::kA>(T), _activeW+1, SINGLESTEP, &z, NOSTEP, _activeW+1, SINGLESTEP);

                    vsMulI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, _indexW+bpprix::decode<bpprix::kM>(T)+1, NOSTEP, _indexW+1, SINGLESTEP);
                    for (unsigned short ix = 0; ix < nActive; ++ix) {
                        vsAddI(bpprix::decode<bpprix::kM>(T)-_pmix[ix]+1,  _indexW+_pmix[ix]+1, SINGLESTEP, &ONEf, NOSTEP, _indexW+_pmix[ix]+1, SINGLESTEP);
                    }
                    z = 1 / _indexW[bpprix::decode<bpprix::kM>(T)+1];
                    vsMulI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, &z, NOSTEP, _indexW+1, SINGLESTEP);
                    LOGVER("[ZellnerSiow::propose] Birth Adapted Signal Cumulative-Weighting %u.%u", prec::_XX, _indexW[1]); 
                    for (unsigned short ix = 1; ix <= bpprix::decode<bpprix::kM>(T); ++ix) { APPEND(VERBOSE, "%u.%u", prec::_XX, _indexW[ix+1]); }

                    coefs(_projesky.k()-1); // Log the new direction's coefficients
                } else {
                    LOGVER("[ZellnerSiow::propose] Birth Proposal Rejected with log(MH) %n.%u", prec::_XX, mhr); 
                    _knots.set(_projesky.k(), _dscr.kKnotQuNScale);
                }
            }
        }

        template<>
        void propose<action::kDeath>() {
            unsigned short dix = 0;
            _knots.runif(static_cast<unsigned short>(0), static_cast<unsigned short>(_projesky.k()), dix);
            const unsigned short nActive = _knots.index(dix, _pmix);
            LOGVER("[ZellnerSiow::propose] Death Proposal Index %u with nActive %u", prec::_, dix, nActive); 
            _mh[mhix::kSelection] = ((_activeW[nActive] - _activeW[nActive-1]) * _activeW[bpprix::decode<bpprix::kA>(T)+1] - 1);
            _mh[mhix::kSelection] = std::log(_mh[mhix::kSelection] / (_activeW[bpprix::decode<bpprix::kA>(T)+1] - 1));

            if (nActive > 1) {
                // Modify the proposal _indexW. If the proposal is rejected, will need to undo
                vsMulI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, _indexW+bpprix::decode<bpprix::kM>(T)+1, NOSTEP, _indexW+1, SINGLESTEP);
                for (unsigned short ix = 0; ix < nActive; ++ix) { 
                    vsSubI(bpprix::decode<bpprix::kM>(T)-_pmix[ix]+2,  _indexW+_pmix[ix], SINGLESTEP, &ONEf, NOSTEP, _indexW+_pmix[ix], SINGLESTEP);
                }
                const float z = 1 / _indexW[bpprix::decode<bpprix::kM>(T)+1];
                vsMulI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, &z, NOSTEP, _indexW+1, SINGLESTEP);
                LOGVER("[ZellnerSiow::propose] Death Pre-Adapted Signal Cumulative-Weighting %u.%u", prec::_XX, _indexW[1]); 
                for (unsigned short ix = 1; ix <= bpprix::decode<bpprix::kM>(T); ++ix) { APPEND(VERBOSE, "%u.%u", prec::_XX, _indexW[ix+1]); }
    
                _mh[mhix::kSelection] += (lnCr(bpprix::decode<bpprix::kM>(T), nActive) + std::log(dwallenius(nActive)));
            }
            contractrfpf<1, update::kFull>(_projesky, dix); 
            _mh[mhix::kPrior] = 2*std::log(_projesky.template dset<factor::kProposal>() / _projesky.det());
            LOGVER("[ZellnerSiow::propose] Death Proposal Anti-Similarity Ratio %n.%u", prec::_XX, _mh[mhix::kPrior]);

            const MKL_INT ex = dix*(kD-2)+bpprix::decode<bpprix::kI>(T);
            contractrfpf<kD-2, update::kFull>(_basesky, ex);
            bool execd = false;
            float pse = beta<action::kDeath>(ex);
            if (pse < _tau*_ssy) {
                _mh[mhix::kProposal] = getMH<mhix::kSpageiria>(_projesky.k()-1);  
                pse = _ssy - pse;
                _mh[mhix::kTotal] = _mh[mhix::kSpageiria] - _mh[mhix::kProposal] + _mh[mhix::kSelection] + _mh[mhix::kPrior]
                                                    - N*(std::log(pse) - std::log(_sse))/2 + std::log(_projesky.k()/_dscr.kLambda) + (kD-2)*std::log(1+_tau)/2;
                float mhr = 0;
                _knots.runif(0.0f, 1.0f, mhr);
                mhr = std::log(mhr);
                LOGVER("[ZellnerSiow::propose] Death Proposal SSE %u.%u", prec::_X, pse); APPEND(VERBOSE, "vs Control SSE %u.%u", prec::_X, _sse);
                if (mhr < _mh[mhix::kTotal]) { // Then the proposal is accepted
                    execd = true;
                    LOGLAC("[ZellnerSiow::propose] Death ACCEPTED with %n.%u", prec::_XX, mhr); APPEND(LACONIC, "< %n.%u", prec::_XX, _mh[mhix::kTotal]);
                    _knots.del(dix, _projesky.k(), _dscr.kKnotQuNScale);
                    const unsigned short pex = ex+kD-2;
                    const unsigned short n = (_basesky.k()-pex)*sizeof(float);
                    std::memmove(_basis+N*ex, _basis+N*pex, N*n);
                    std::memmove(_by+ex, _by+pex, n);
                    _projesky.accept();
                    _basesky.accept();

                    _sxy = _ssy - pse;
                    _sse = pse;
                    _mh[mhix::kSpageiria] = _mh[mhix::kProposal];
                    std::memcpy(_beta, _working, _basesky.k()*sizeof(float));
                } else {
                    LOGVER("[ZellnerSiow::propose] Death Proposal Rejected with log(MH) %n.%u", prec::_XX, mhr); 
                }
            }
            if (!execd && nActive > 1) {
                // Undo the change to the active & index weights
                vsMulI(bpprix::decode<bpprix::kA>(T), _activeW+1, SINGLESTEP, _activeW+bpprix::decode<bpprix::kA>(T)+1, NOSTEP, _activeW+1, SINGLESTEP);
                vsAddI(bpprix::decode<bpprix::kA>(T)-nActive+2,  _activeW+nActive, SINGLESTEP, &ONEf, NOSTEP, _activeW+nActive, SINGLESTEP);
                float z = 1 / _activeW[bpprix::decode<bpprix::kA>(T)+1];
                vsMulI(bpprix::decode<bpprix::kA>(T), _activeW+1, SINGLESTEP, &z, NOSTEP, _activeW+1, SINGLESTEP);

                vsMulI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, _indexW+bpprix::decode<bpprix::kM>(T)+1, NOSTEP, _indexW+1, SINGLESTEP);
                for (unsigned short ix = 0; ix < nActive; ++ix) { // NOTE: _pmix is still one-indexed here
                    vsAddI(bpprix::decode<bpprix::kM>(T)-_pmix[ix]+2,  _indexW+_pmix[ix], SINGLESTEP, &ONEf, NOSTEP, _indexW+_pmix[ix], SINGLESTEP);
                }
                z = 1 / _indexW[bpprix::decode<bpprix::kM>(T)+1];
                vsMulI(bpprix::decode<bpprix::kM>(T), _indexW+1, SINGLESTEP, &z, NOSTEP, _indexW+1, SINGLESTEP);
                LOGVER("[ZellnerSiow::propose] Death Undone Signal Cumulative-Weighting %u.%u", prec::_XX, _indexW[1]); 
                for (unsigned short ix = 1; ix <= bpprix::decode<bpprix::kM>(T); ++ix) { APPEND(VERBOSE, "%u.%u", prec::_XX, _indexW[ix+1]); }
            }
        }

        template<>
        void propose<action::kChange>() {
            unsigned short rix = 0;
            _knots.runif(static_cast<unsigned short>(0), static_cast<unsigned short>(_projesky.k()), rix);
            const unsigned short nActive = _knots.index(rix, _pmix, _pmu);
            LOGVER("[ZellnerSiow::propose] Change Proposal Index %u with nActive %u", prec::_, rix, nActive); 
            // Return the recovered signal indices to zero-base after one-base-oriented indexing functions
            for (unsigned short ix = 0; ix < nActive; ++ix) { _pmix[ix] -= 1; }

            if (nActive > 1) {
                LOGVER("[ZellnerSiow::propose] Change Prior Direction %n.%u", prec::_XX, _pmu[0]);
                for (unsigned short ix = 1; ix < nActive; ++ix) { APPEND(VERBOSE, "%n.%u", prec::_XX, _pmu[ix]); } 
                _knots.rps(_pmu, _dscr.kSphericalK);
                LOGVER("[ZellnerSiow::propose] Change Proposal Direction %n.%u", prec::_XX, _pmu[0]);
                for (unsigned short ix = 1; ix < nActive; ++ix) { APPEND(VERBOSE, "%n.%u", prec::_XX, _pmu[ix]); } 
                toBasis(_projesky.k(), _knots.get(_projesky.k(), _pmix, _pmu, _knots.rsnormal(_dscr.kKnotQuNScale)));
            } else {
                toBasis(_projesky.k(), _knots.get(_projesky.k(), _pmix[0], _knots.rbool(), _knots.rsnormal(_dscr.kKnotQuNScale))); 
            }
            modifyrfpf<1, update::kFull>(_projesky, rix); 
            _mh[mhix::kPrior] = 2*std::log(_projesky.template dset<factor::kProposal>() / _projesky.det());
            LOGVER("[ZellnerSiow::propose] Change Proposal Anti-Similarity Ratio %n.%u", prec::_XX, _mh[mhix::kPrior]);
            if (std::isinf(_mh[mhix::kPrior]) || std::isnan(_mh[mhix::kPrior])) {
                LOGLAC("[ZellnerSiow::propose] High Proposal Similarity %n.%u. Abort this change proposal.", prec::_X, _mh[mhix::kPrior]);
                return;
            }

            const MKL_INT r = rix*(kD-2)+bpprix::decode<bpprix::kI>(T);
            modifyrfpf<kD-2, update::kFull>(_basesky, r);
            float pse = beta<action::kChange>(r);
            if (pse < _tau*_ssy) {
                pse = _ssy - pse;
                _mh[mhix::kTotal] = -N*(std::log(pse) - std::log(_sse))/2 + _mh[mhix::kPrior];
                float mhr = 0;
                _knots.runif(0.0f, 1.0f, mhr);
                mhr = std::log(mhr);
                LOGVER("[ZellnerSiow::propose] Change Proposal SSE %u.%u", prec::_X, pse); APPEND(VERBOSE, "vs Control SSE %u.%u", prec::_X, _sse);
                if (mhr < _mh[mhix::kTotal]) { // Then the proposal is accepted
                    LOGLAC("[ZellnerSiow::propose] Change ACCEPTED with %n.%u", prec::_XX, mhr); APPEND(LACONIC, "< %n.%u", prec::_XX, _mh[mhix::kTotal]);
                    _knots.mod(rix, _projesky.k());
                    std::memcpy(_basis+N*r, _basis+N*_basesky.k(), N*(kD-2)*sizeof(float));
                    std::memcpy(_by+r, _working+_basesky.k(), (kD-2)*sizeof(float));
                    _projesky.accept();
                    _basesky.accept();
                    _sxy = _ssy - pse;
                    _sse = pse;
                    std::memcpy(_beta, _working, _basesky.k()*sizeof(float));
                } else {
                    LOGVER("[ZellnerSiow::propose] Change Proposal Rejected with log(MH) %n.%u", prec::_XX, mhr); 
                } 
            }
            _knots.set(_projesky.k(), _dscr.kKnotQuNScale);
        }

      /**
        * @brief The structure of either Basis or Projection type, that holds both the control and proposal
        *              RFPF cholesky factors
        */
        template<unsigned short L, unsigned short M> struct Factor<L, M> {
            static constexpr unsigned short kAlloc = (M * (M + 1)) / 2;
            static constexpr unsigned short kSrc    = L*M;
            static constexpr MKL_INT kN                  = L;
            static constexpr MKL_INT kM                 = M;

            struct RFPF {
                bool odd;
                MKL_INT ki;
                MKL_INT m1, m2;
                float* const lo;
                float* up;

                RFPF() : lo(_cholesky+1), _det(1) {}

                float(&get())[kAlloc] { return  _cholesky; }
                const MKL_INT& k() const { return _k; }
                void set(const MKL_INT& k, const float& det) {
                    _k = k;
                    odd = k%2;
                    ki = k + 1 + odd;
                    m1 = (ki - 1) / 2;
                    m2 = k - m1;
                    up = odd ? _cholesky+ki+1 : _cholesky;
                    _det = det;
                }
                const float& det() const { return _det; }
                const float dset() {
                    _det = *lo;
                    for (unsigned short ix = 1; ix < m1; ++ix) { _det *= lo[ix*(ki+1)]; }
                    for (unsigned short ix = 0; ix < m2; ++ix) { _det *= up[ix*(ki+1)]; }
                    return _det;
                }

              private:
                MKL_INT _k;
                float _det;
                float _cholesky[kAlloc];
            };

            Factor(const float(&src)[kSrc]) : _src(src) {}

            const float(&src())[kSrc] { return  _src; }
            void accept() { 
                set(k<factor::kProposal>(), det<factor::kProposal>()); 
                std::memcpy(get(), get<factor::kProposal>(), _control.ki*_control.m1*sizeof(float)); 
            }

            template<factor_t... F> float(&get())[kAlloc] { return Impl<F...>::get(*this); }
            template<factor_t... F> const MKL_INT& k() const { return Impl<F...>::k(*this); }
            template<factor_t... F> void set(const MKL_INT& k, const float& det = 1) {  Impl<F...>::set(*this, k, det); }
            template<factor_t... F> const bool odd() const { return Impl<F...>::odd(*this); }
            template<factor_t... F> const RFPF& rfpf() const { return Impl<F...>::rfpf(*this); }
            template<factor_t... F> const float det() const { return Impl<F...>::det(*this); }
            template<factor_t... F> const float dset() { return Impl<F...>::dset(*this); }

          private:
            template<factor_t... F> struct Impl {
                static float(&get(Factor<L, M>& f))[kAlloc] { return f._control.get(); }
                static const MKL_INT& k(const Factor<L, M>& f) { return f._control.k(); }
                static void set(Factor<L, M>& f, const MKL_INT& k, const float& det) { f._control.set(k, det); }
                static const bool odd(const Factor<L, M>& f) { return f._control.odd; }
                static const RFPF& rfpf(const Factor<L, M>& f) { return f._control; }
                static const float det(const Factor<L, M>& f) { return f._control.det(); }
                static const float dset(Factor<L, M>& f) { return f._control.dset(); }
            };
            template<> struct Impl<factor::kProposal> {
                static float(&get(Factor<L, M>& f))[kAlloc] { return f._proposal.get(); }
                static const MKL_INT& k(const Factor<L, M>& f) { return f._proposal.k(); }
                static void set(Factor<L, M>& f, const MKL_INT& k, const float& det) { f._proposal.set(k, det); }
                static const bool odd(const Factor<L, M>& f) { return f._proposal.odd; }
                static const RFPF& rfpf(const Factor<L, M>& f) { return f._proposal; }
                static const float det(const Factor<L, M>& f) { return f._proposal.det(); }
                static const float dset(Factor<L, M>& f) { return f._proposal.dset(); }
            };
            const float(&_src)[kSrc];
            RFPF _control;
            RFPF _proposal;
        };
        template<factor_t F> struct Factor<F>; 
        template<> struct Factor<factor::kBasis> : public Factor<N, RK> {};
        template<> struct Factor<factor::kProj> : public Factor<bpprix::decode<bpprix::kM>(T), bpprix::decode<bpprix::kR>(T)> {};  

        const Descriptor<T> _dscr;
        Knots<T> _knots;

        float _basis[NR];  // Need this for assessing new ridge functions and potentially queueing obervations to roll-on & -off
        float _prsis[PRK]; // The basis expansion for single-sample regressor predictions
        Factor<factor::kBasis> _basesky;
        Factor<factor::kProj> _projesky;
        float _by[RK];       // The basis-transformed response vector B^T.y
        float _beta[RK];   // The current basis regression coefficient vector
        std::pair<unsigned short[P], float[PRK]> _cache;
        float _tau, _sigma;
        float _ssy, _sse, _sxy;
        float _y[N];
        float _working[N];
        float _activeW[bpprix::decode<bpprix::kA>(T)+2]; // The adaptive weights for the number of active signals to choose 
        float _indexW[bpprix::decode<bpprix::kM>(T)+2]; // The adaptive weights for the signal indices to choose 
        unsigned short _pmix[bpprix::decode<bpprix::kA>(T)];
        float _pmu[bpprix::decode<bpprix::kA>(T)];
        float _mh[mhix::N];
        MKL_INT _outcome;
    };

 } // namespace bppr
