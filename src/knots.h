/**
 * *****************************************************************************
 * \file knots.h
 * \author Graham Beck
 * \brief BPPR: Maintains the knots used for spline bases
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include "logging.h"
#include "projection.h"
#include "mkl_vml.h"


namespace bppr
{
    template <bppr_t T>
    class Knots
    {
      public:
        static constexpr float kQEpsilon        = cx::pow(IGOLDENRATIO, 8); // Accuracy of the streaming quantiles

        static constexpr MKL_INT kD            = bpprix::decode<bpprix::kK>(T) + 2; // Number of spline points
        static constexpr MKL_INT N              = bpprix::decode<bpprix::kN>(T);       // Number of training samples
        static constexpr unsigned short R   = bpprix::decode<bpprix::kR>(T);        // Number of ridge functions = number of nu values

        using gettable_t = std::pair<const float(&)[kD], const float(&)[N]>;
        template<unsigned short P>
        using predable_t = std::pair<const float(&)[kD*R*P], const float(&)[R*P]>;

       /**
        * @brief Knots primarily transforms the linear projection of the data given by Projection 
        *              by sigmoidally/tanhally compressing/rarefying the quantiles given parameter nu, 
        *              before finding those quantiles of the actual projections. 
        */
        Knots(const Descriptor<T>& dscr, Proto<T>& proto) 
            : Knots(dscr, proto, RawQ<kD>().run(dscr.kKnotQrZero), std::make_index_sequence<kD>{}) {}

        ~Knots()
        {
            vslDeleteStream(&_randomStream);
            _qStatus = vslSSDeleteTask(&_qTask);
        }

        /**
        * @brief The canonical get for the class, returning either its main feature: the knots matrix itself, 
        *              or acting as a pass-through for Projection, to return the transform matrix 
        */
        template<typename U=Knots<T>>
        const auto& get(){
            return _knots;
        }
        template<>
        const auto& get<Projection<T>>() {
            return _projection.get();
        }
        
      /**
        * @brief Package the projection along with the knots for ridge ix
        */
        gettable_t get(const unsigned short ix) {
            _projection.get(ix, _working);
            return {reinterpret_cast<float(&)[kD]>(_knots[ix*kD]), _working};
        }
        /**
        * @brief Get the knots+projection package for new single-signal ridge rix
        */
        gettable_t get(const unsigned short rix, const unsigned short mix, const bool negated, const float nu) {
            _projection.set(rix, mix, negated, _working);
            if (nu != 0) { _nu[rix] = nu; sigmoidize(rix);}
            quantize(rix);
            return {reinterpret_cast<float(&)[kD]>(_knots[rix*kD]), _working};
        }
        gettable_t get(const unsigned short rix, const unsigned short mix, const bool negated) {
            return get(rix, mix, negated, 0);
        }
        /**
        * @brief Get the knots+projection package for new multi-signal ridge rix with signal indices/vals given by mix/in.
        * 
        * @note Sigmoidize is OFF when nu=0 for what are assumed here to be new projection proposals, as it should have already
        *              been initialized at nu[rix]. Re-sigmoidization with a new nu occurs on kDeath so that knotsx is maintained with 
        *              actual quantile thresholds rather than the data quantiles.  
        */
        template<unsigned short M>
        gettable_t get(const unsigned short rix, const unsigned short(&mix)[M], const float(&in)[M], const float nu) {
            _projection.set(rix, mix, in, _working);
            if (nu != 0) {  _nu[rix] = nu; sigmoidize(rix); }
            quantize(rix);
            return {reinterpret_cast<float(&)[kD]>(_knots[rix*kD]), _working};
        }
        template<unsigned short M>
        gettable_t get(const unsigned short rix, const unsigned short(&mix)[M], const float(&in)[M]) {
            return get(rix, mix, in, 0);
        }

        /**
        * @brief The form of get for prediction using all P cached state segments. 
        *              Return a package of all cached knots and the projection of X given all cached ridge functions
        */
        predable_t<bpprix::decode<bpprix::kP>(T)> get(const MKL_UINT(&rcx)[rcx::N], const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            _projection.get(rcx, X, reinterpret_cast<float(&)[PR]>(_working));
            return {_cache, reinterpret_cast<float(&)[PR]>(_working)};
        }

       /**
        * @brief The form of prediction-oriented get for updating. Only return the current knots and current 
        *              projection-of-X package, but replace for isx of Projection's data matrix _X with new sample X
        */
        predable_t<1> get(const unsigned short isx, const MKL_INT& nRidge, const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            _projection.get(isx, nRidge, X, reinterpret_cast<float(&)[R]>(_working));
            return {_knots, reinterpret_cast<float(&)[R]>(_working)};
        }
        /**
        * @brief Prediction-oriented get using current knots & projection transformations. 
        *              No update to Projection's data matrix _X
        */
        predable_t<1> get(const MKL_INT& nRidge, const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            _projection.get(nRidge, X, reinterpret_cast<float(&)[R]>(_working));
            return {_knots, reinterpret_cast<float(&)[R]>(_working)};
        }

        /**
        * @brief Sigmoidize the raw quantiles at ridge rix, given a standard deviation measure for nu. 
        */
        void set(const unsigned short rix, const float& stddev) {
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, _randomStream, 1, _nu+rix, 0, stddev);
            sigmoidize(rix);
        }

       /**
        * @brief Pass-through to Projection of the multiplicity-of-signal-combination test.
        */
        bool has(const unsigned short test, const unsigned short nRidge, unsigned short multiplicity=1) {
            return _projection.has(test, nRidge, multiplicity);
        }

       /**
        * @brief Pass-through to Projection of the request for the projection of ridge ix. 
        */
        const float(&projection(const unsigned short ix))[N] {
            _projection.get(ix, _working);
            return _working;
        }

      /**
        * @brief Removes a ridge slot, pulling all subsequent knots and nu parameters into the gap. 
        */
        void del(const unsigned short rix, const unsigned short nRidge, const float& stddev) {
            _projection.del(rix, nRidge);
            const unsigned short n = (nRidge-rix-1)*sizeof(float);
            std::memmove(_nu+rix, _nu+rix+1, n);
            std::memmove(_knots+rix*kD, _knots+(rix+1)*kD, n*kD);
            // Replace the exposed _nu and _knotsx at index nRidge-1 with new random pre-quantile values
            set(nRidge-1, stddev);
        }

      /**
        * @brief Used by an accepted 'change' action, the proposed knots and nu at index nRidge
        *              overwrite the previous ones at rix.  
        */
        void mod(const unsigned short rix, const unsigned short nRidge) {
            _projection.mod(rix, nRidge);
            LOGVER("[Knots::mod] Replacing nu %n.%u -> ", prec::_XX, _nu[rix]); APPEND(VERBOSE, "%n.%u", prec::_XX, _nu[nRidge]);
            _nu[rix] = _nu[nRidge];
            std::memcpy(_knots+rix*kD, _knots+nRidge*kD, kD*sizeof(float));
        }

      /**
        * @brief Take the sigmoidized quantiles and apply them to the projection for ridge ix, to give the data quantiles. 
        * 
        * @note The (modified) quantiles desired must already have been calculated and inserted into _knots BEFORE this call 
        *              and _working must have already been populated with the projection
        */
        void quantize(const unsigned short rix) {
            float* knotsx = _knots+rix*kD;
            LOGVER("[Knots::quantize] Raw Quantiles(%u) : ", prec::_, rix, L3_ARG_UNUSED); 
            for (unsigned short ix = 0; ix < kD; ++ix) { APPEND(VERBOSE, "%n.%u", prec::_XX, knotsx[ix]); }
            _qStatus = vslsSSEditStreamQuantiles(_qTask, &kD, knotsx, knotsx, &QPTYPE, &kQEpsilon);
            _qStatus = vslsSSCompute(_qTask, VSL_SS_STREAM_QUANTS, VSL_SS_METHOD_SQUANTS_ZW);
            LOGVER("[Knots::quantize] Data Quantiles(%u) : ", prec::_, rix, L3_ARG_UNUSED); 
            for (unsigned short ix = 0; ix < kD; ++ix) { APPEND(VERBOSE, "%n.%u", prec::_XX, knotsx[ix]); }
        }

      /**
        * @brief Copies the knots into a circular buffer and updates its head and tail  
        */
        void cache(MKL_UINT(&rcx)[rcx::N], const unsigned short nRIn, const unsigned short nROut) {
            _projection.cache(rcx, nRIn, nROut);
            unsigned short n = std::min(static_cast<MKL_UINT>(nRIn), PR-rcx[rcx::kKHx]);
            LOGVER("[Knots::cache] Caching ix %u +> %u", prec::_, rcx[rcx::kKHx], n);
            std::memcpy(_cache+rcx[rcx::kKHx]*kD, _knots, n*kD*sizeof(float));

            // Wrap around if necessary
            n = nRIn - n;
            if (n > 0) { std::memcpy(_cache, _knots+(nRIn-n)*kD, n*kD*sizeof(float)); }

            rcx[rcx::kKHx] += nRIn; rcx[rcx::kKHx] %= PR;
            if (rcx[rcx::kPn] >= bpprix::decode<bpprix::kP>(T)) {
                rcx[rcx::kKTx] += nROut; rcx[rcx::kKTx] %= PR;
            }
            LOGVER("[Knots::cache] kKTx -> %u, kKHx -> %u", prec::_, rcx[rcx::kKTx], rcx[rcx::kKHx]);
        }

      /**
        * @brief Pass-throughs of various convenience forms of the cosine measures of similarity 
        *              between the transformation at rix and the n transforms starting at rsx 
        */
        void cos(const MKL_INT rix, float(&out)[bpprix::decode<bpprix::kR>(T)]) {
            _projection.cos(0, rix+1, rix, out, SINGLESTEP, ZEROf);
        }
        void cos(const MKL_INT rsx, const MKL_INT n, const MKL_INT rix, float(&out)[bpprix::decode<bpprix::kR>(T)]) {
            _projection.cos(rsx, n, rix, out, SINGLESTEP, ZEROf);
        }
        void cos(const MKL_INT rsx, const MKL_INT n, const MKL_INT rix, float* out, const MKL_INT incr, const float beta) {
            _projection.cos(rsx, n, rix, out, incr, beta);
        }

      /**
        * @brief Returns an inverse-gamma distributed random variable given distribution parameters alpha & beta
        */
        void rigamma(const float alpha, const float beta, float& out) {
            vsRngGamma(VSL_RNG_METHOD_GAMMA_GNORM, _randomStream, 1, &out, alpha, 0, beta);
            out = 1 / out;
        }

      /**
        * @brief Returns an beta distributed random variable given distribution parameters alpha & beta
        */
        void rbeta(const float alpha, const float beta, float& out) {
            vsRngBeta(VSL_RNG_METHOD_BETA_CJA, _randomStream, 1, &out, alpha, beta, 0, 1);
        }

      /**
        * @brief Returns an array of k classic N(0,1) normal random normal variables
        */
        template<unsigned short M>
        void rsnormal(const unsigned short k, float(&out)[M]) {
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, _randomStream, k, out, 0, 1);
        }
      /**
        * @brief Returns a single normal random variable with specified standard deviation
        */
        float rsnormal(const float& stddev) {
            float out = 0;
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, _randomStream, 1, &out, 0, stddev);
            return out;
        }

      /**
        * @brief Returns a uniformly-distributed random variable in the interval [lb,ub]
        */
        template<typename U>
        void runif(const U lb, const U ub, U& out) {
            vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _randomStream, 1, &out, lb, ub);
        }
        template<>
        void runif<unsigned short>(const unsigned short lb, const unsigned short ub, unsigned short& out) {
            viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _randomStream, 1, reinterpret_cast<MKL_INT*>(&out), lb, ub);
        }

      /**
        * @brief Returns a random boolean. 
        */
        const bool rbool() {
            int out = 0;
            viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _randomStream, 1, &out, 0, 2);
            return static_cast<bool>(out);
        }

      /**
        * @brief Returns in array mu a random M-dimensional sample from the 
        *              Power Spherical distribution with concentration parameter kappa
        */
        template<unsigned short M>
        void rps(float(&mu)[M], const float kappa) {
            constexpr MKL_INT L = M-1;
            constexpr MKL_INT m = M;
            _psu[0] = 1; std::memset(_psu+1, 0, L*sizeof(float)); 
            saxpy(&m, &NEGATIVEONEf, mu, &SINGLESTEP, _psu, &SINGLESTEP);
            float x = 1 / snrm2 (&m, _psu, &SINGLESTEP);
            sscal(&m,  &x, _psu, &SINGLESTEP);
            
            x = static_cast<float>(L)/2;
            rbeta(x+kappa, x, x);
            mu[0] = 2*x-1;
            
            rsnormal(L, reinterpret_cast<float(&)[L]>(mu[1]));
            x = 1 / snrm2 (&L, mu+1, &SINGLESTEP);
            sscal(&L,  &x, mu+1, &SINGLESTEP);
            x = std::sqrt(1 - mu[0]*mu[0]);
            sscal(&L,  &x, mu+1, &SINGLESTEP);
            x = -2*sdot(&m, mu, &SINGLESTEP, _psu, &SINGLESTEP);
            saxpy(&m, &x, _psu, &SINGLESTEP, mu, &SINGLESTEP);
        }

      /**
        * @brief Pass-through of the signal index tracking functionality in Projection
        */
        template<typename... Ts>
        unsigned short index(Ts&... args) { return _projection.index(args...); }
        template<> unsigned short index<const unsigned short>(const unsigned short& rix) { return _projection.index(rix); }

      /**
        * @brief Persist state to file. 
        */
        void write(Proto<T>& proto, const unsigned short nRidge, MKL_UINT(&rcx)[rcx::N]) {
            _projection.write(proto, nRidge, rcx);
            rectify<kD, PR>(_cache, rcx[rcx::kKHx], rcx[rcx::kKTx]);
            rcx[rcx::kKTx] = 0; rcx[rcx::kKHx] = rcx[rcx::kRx];
            proto.template set<proto::kKnot>(nRidge, _nu, rcx[rcx::kRx], _cache);
        }

      private:
        static constexpr unsigned short PR   = bpprix::decode<bpprix::kP>(T) * R;

        template<std::size_t... iK>
        Knots(const Descriptor<T>& dscr, Proto<T>& proto, const float(&rawQuantiles)[kD], std::index_sequence<iK...>) 
            : _projection(dscr, proto)
            , _rawQuantiles{(*(rawQuantiles + iK))...}
        {
            vslNewStream(&_randomStream, VSL_BRNG_MT19937, dscr.rSeed);

            // Draw random gaussian nus for the quantile transformations for all possible ridge functions
            vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, _randomStream, R, _nu, 0.0, dscr.kKnotQuNScale);

            // Set up the estimate-quantiles task
            _qStatus = vslsSSNewTask(&_qTask, &SINGLESTEP, &N, &ORIENTATION, _working, 0, 0);

            // Overwrite the first nRidge nu when we have initialization information
            unsigned short nRidge = 0;
            proto.template get<proto::kKnot>(nRidge, _nu, _cache);

            // Use the nu values to transform the raw quantiles to sigmoidal quantiles for each ridge function
            for (unsigned int rix = 0; rix < R; ++rix) {
                sigmoidize(rix);
                if (rix < nRidge) {
                    projection(rix);  // Populate the working array with this ridge's projection
                    quantize(rix);
                }
            }
        }
        
      /**
        * @brief Compress or rarefy the raw quantiles at ridge rix, given 
        *              corresponding parameter nu that must already have been set. 
        * 
        * @details More positive values of nu force the quantiles towards the extremes, 
        *                  concentrating the knots at higher and lower quantiles. More negative
        *                  values bring the quantiles' concentration closer to the median. 
        */
        void sigmoidize(const unsigned short rix) {
            float* knotsx = _knots+rix*kD;
            vsInv(kD, _rawQuantiles, knotsx);
            vsSubI(kD, knotsx, SINGLESTEP, &ONEf, NOSTEP, knotsx, SINGLESTEP);
            vsPowx(kD, knotsx, -std::exp(_nu[rix]), knotsx);
            vsAddI(kD, knotsx, SINGLESTEP, &ONEf, NOSTEP, knotsx, SINGLESTEP);
            vsInv(kD, knotsx, knotsx);
            vsSubI(kD, &ONEf, NOSTEP, knotsx, SINGLESTEP, knotsx, SINGLESTEP);
        }

        VSLStreamStatePtr _randomStream;
        const float _rawQuantiles[kD];
        float _nu[R];                 // The sigmoidal transformation parameter. Larger than zero -> Quantiles bulge outwards 
        float _knots[R * kD];   // Knot values, organized as R lots of kD-arrays, the sigmoidal quantiles for each ridge function
        float _cache[PR * kD];
        float _working[N]; 
        float _psu[bpprix::decode<bpprix::kA>(T)]; 
        int _qStatus;
        VSLSSTaskPtr _qTask;

        Projection<T> _projection;
    };

} // namespace bppr