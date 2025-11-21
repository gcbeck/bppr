/**
 * *****************************************************************************
 * \file bppr.h
 * \author Graham Beck
 * \brief BPPR: Bayesian Projection Pursuit Regression, based on the work by Collins et al
 *                        at https://github.com/gqcollins/BayesPPR.
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include "zellner.h"


namespace bppr
{
    template <bppr_t T>
    class BPPR
    {
      public:
        using state_t = unsigned short; 

        static constexpr state_t kAdapt = 0x0000;
        static constexpr state_t kBurn   = 0x0001;
        static constexpr state_t kPost    = 0x0002;

       /**
        * @brief BPPR controls the iteration through MCMC phases to perform the training, 
        *              and facilitates the subequent updating and/or predicting. 
        */
        template<size_t M>
        BPPR(const Descriptor<T>& dscr, const std::array<char, M>& from)
            : _proto(from)
            , _zs(dscr, _proto)
            , _nEvery(dscr.nEvery)
        {
            if (!_proto.template get<proto::kCache>(_rcx)) {
                std::fill(_rcx, _rcx+rcx::N, 0);
                _rcx[rcx::kZHx] = _rcx[rcx::kZTx] = bpprix::decode<bpprix::kI>(T) ? bpprix::decode<bpprix::kP>(T) : 0;
            }
            _proto.close();

            for (unsigned short ix = 0; ix < dscr.nAdapt; ++ix, ++_rcx[rcx::kMCx]) {
                LOGLAC("[BPPR::ctor] ADAPTIVE MCMC Iteration %u / %u", prec::_, ix, dscr.nAdapt);
                _zs.template mcmc<kAdapt>(_rcx);
            }
            for (MKL_UINT ix = 0; ix < dscr.nBurn; ++ix, ++_rcx[rcx::kMCx]) {
                LOGLAC("[BPPR::ctor] BURN MCMC Iteration %u / %u", prec::_, ix, dscr.nBurn);
                _zs.template mcmc<kBurn>(_rcx); 
            }
            const unsigned nPost =  (dscr.P-1)*dscr.nEvery+1;
            for (unsigned short ix = 0; ix < nPost; ++ix, ++_rcx[rcx::kMCx]) {
                if (ix % dscr.nEvery == 0) {
                    LOGLAC("[BPPR::ctor] POSTERIOR Cached MCMC Iteration %u / %u", prec::_, ix, nPost);
                    _zs.template mcmc<kPost>(_rcx); 
                } else {
                    LOGLAC("[BPPR::ctor] POSTERIOR Noncached MCMC Iteration %u / %u", prec::_, ix, nPost);
                    _zs.template mcmc<kBurn>(_rcx); 
                }
            }

            _zs.coefs();  // Log the direction coefficients over all ridge functions
            for (unsigned short ix = 0; ix < bpprix::decode<bpprix::kN>(T); ix+=16) { // DEBUGGING TODO: Delete
                _zs.debugX(_rcx, ix);
            }
        }

      /**
        * @brief Introduce a new sample (X,y) to update the BPPR state with. 
        * 
        * @details Alongside the roll-on/off of basis-transformed samples and update of the basis cholesky matrix, 
        *                 a ridge birth/death/change proposal is made and accepted or rejected and the Zellner-Siow 
        *                 shrinkage and variance hyperparameters updated. Every 'nEvery' iterations the sampled regression
        *                 vector is cached to update the posterior set of bppr samples. 
        */
        void update(const float(&X)[bpprix::decode<bpprix::kM>(T)], const float y) {
            _zs.update(X, y, _rcx); 
            if (_rcx[rcx::kMCx] % _nEvery == 0) {
                LOGLAC("[BPPR::update] UPDATE Cached MCMC Iteration %u", prec::_, _rcx[rcx::kMCx], L3_ARG_UNUSED);
                _zs.template mcmc<kPost>(_rcx); 
            } else {
                LOGLAC("[BPPR::update] UPDATE Noncached MCMC Iteration %u", prec::_, _rcx[rcx::kMCx], L3_ARG_UNUSED);
                _zs.template mcmc<kBurn>(_rcx); 
            }
            ++_rcx[rcx::kMCx];
        }
      /**
        * @brief Convenience/demonstration function that performs all online updates from the OOS rows in file 'from'. 
        */
        template<size_t M>
        void update(const std::array<char, M>& from) { 
            _proto.template open<M, proto::kFUpdate>(from);
            float X[bpprix::decode<bpprix::kM>(T)]; float y;
            while(_proto.template get<proto::kFUpdate>(X, y)) {
                update(X, y);
            }
            _proto.close();
        }

      /**
        * @brief Predict the response to signals input array X.   
        */
        template<pred_t U>
        float predict(const float(&X)[bpprix::decode<bpprix::kM>(T)]) {
            return _zs.template predict<U>(_rcx, X);
        }
      /**
        * @brief Convenience/demonstration function that performs all online predictions
        *              from the OOS rows in file 'from' (which includes the supervision 'true' y values, unused here)
        */
        template<pred_t U, size_t M>
        void predict(const std::array<char, M>& from) { 
            _proto.template open<M, proto::kFUpdate>(from);
            float X[bpprix::decode<bpprix::kM>(T)]; float y;
            constexpr MKL_INT P = bpprix::decode<bpprix::kP>(T);
            while(_proto.template get<proto::kFUpdate>(X, y)) {
                const float pred = predict<U>(X);
                LOGLAC("[BPPR::predict] Pred Y %n.%u", prec::_XX, pred); 
                if constexpr(U & pred::kDistribution) {
                    APPEND(LACONIC, "Spread |dY| %u.%u", prec::_XX, sdot(&P, _zs.posterior(pred), &SINGLESTEP, &ONEf, &NOSTEP)/P);
                }
            }
            _proto.close();
        }

      /**
        * @brief Returns the posterior of P predictions
        */
        const float(&posterior())[bpprix::decode<bpprix::kP>(T)] {
            return _zs.posterior();
        }

      /**
        * @brief Persist state to file. 
        */
        template<size_t M>
        void write(const std::array<char, M>& to) {
            _proto.open(to);
            _zs.write(_proto, _rcx);
            _proto.template set<proto::kCache>(_rcx);
            _proto.close();
        }

      private:
        Proto<T> _proto;
        ZellnerSiow<T> _zs;
        const unsigned short _nEvery;
        MKL_UINT _rcx[rcx::N]; // Running indices for mcmc, posterior etc counts, circular buffer indexing
    };

} // namespace bppr