/**
 * *****************************************************************************
 * \file proto.h
 * \author Graham Beck
 * \brief BPPR: Protocol for reading/writing BPPR state to binary file. 
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#pragma once

#include <concepts>
#include <fstream>
#include <stdexcept>

#include "descriptor.h"
#include "util.h"


namespace bppr
{
    using protix_t = unsigned short; // The indices corresponding to the values or offsets of the state variables
    namespace proto {
        // The indices in the file at which is written either the corresponding scalar value or the offset to seek to.
        static constexpr protix_t kNRidge = 0x0000;
        static constexpr protix_t kY            = 0x0001;
        static constexpr protix_t kProj       = 0x0002;
        static constexpr protix_t kKnot      = 0x0003;
        static constexpr protix_t kZS          = 0x0004;
        static constexpr protix_t kRC          = 0x0005;
        static constexpr protix_t kCache    = 0x0006;

        static constexpr protix_t kN           = kCache + 1;

        static constexpr protix_t kFCore      = 0x0010;
        static constexpr protix_t kFUpdate = 0x0020;

        template<protix_t U>
        concept nR_t = (U == kNRidge) || (U == kRC);

        template<protix_t U>
        concept file_t = (U == kFCore) || (U == kFUpdate);
    } // namespace proto

    using rcx_t = unsigned short;
    namespace rcx {
        static const rcx_t kMCx = 0; 
        static const rcx_t kPx    = 1; 
        static const rcx_t kRx    = 2; 
        static const rcx_t kRKx  = 3;
        static const rcx_t kPn    = 4; 
        static const rcx_t kNx    = 5;
        static const rcx_t kPHx  = 6;
        static const rcx_t kPTx   = 7;
        static const rcx_t kKHx  = 8;
        static const rcx_t kKTx   = 9;
        static const rcx_t kZHx  = 10;
        static const rcx_t kZTx   = 11;

        static const rcx_t N         = kZTx + 1;
    } // namespace rcx
    

    template <bppr_t T>
    class Proto{
      public:
        using proto_t  = unsigned int;  // The type of the values themselves
        static constexpr proto_t kAbsent = 0x0000;

      /**
        * @brief The Proto class establishes the persistence protocol, reading and writing to file
        *              for initialization / warmstarting of state. The contructor takes as argument the 
        *              directory where bppr files are persisted. 
        * 
        * @note The expected .bppr file is opened for reading by construction and must be closed 
        *               elsewhere once initialization has been completed. 
        */
        template<size_t M>
        explicit Proto(const std::array<char, M>& from)
            : _from(join<T>(from, suffix<proto::kFCore>()).data(), std::ios::in | std::ios::binary)
        {
            if (!_from.is_open()) {
                throw std::runtime_error(join<T>(from, suffix<proto::kFCore>()).data());
            }
        }

        void close() { _from.close(); }

      /**
        * @brief Opens a binary file for reading or writing, depending on file purpose
        * 
        * @details When open(.) is called explicitly on the .bppr file (signaled by proto::kFCore)
        *                  the file mode is set to write for persistence of the new updated BPPR state. 
        *                  On the other hand when called on the out-of-sample data file (signaled by 
        *                  proto::kFUpdate) then reading is assumed. 
        */
        template<size_t M, protix_t U=proto::kFCore> requires proto::file_t<U>
        void open(const std::array<char, M>& to) { 
            _from.open(join<T>(to, suffix<U>()).data(), mode<U>());
            if (!_from.is_open()) {
                throw std::runtime_error(join<T>(to, suffix<U>()).data());
            }
        }

      /**
        * @brief Reads template-specific data into the arguments submitted
        * 
        * @details The general idea is as follows:  Each proto:: index stores the location 
        *                  of its own data at the beginning of the file, at that index. First we seek
        *                  to that index, read the data location (the offset), then jump to that offset
        *                  to read in the index-specific data. Return true if all expected data was read. 
        *                  The proto::nR_t types are their own offsets, being integer primitives. 
        *                  proto::kCache stores the cached (ring buffered) data for all the other indices. 
        *                  proto::kFUpdate is simpler as there is no offset information in that file, just OOS data. 
        */
        template<protix_t U, typename... Ts>
        bool get(Ts&... args) {
            return io<U>::get(*this, args...);
        }

      /**
        * @brief Writes the protocol-observant data to the file; otherwise similar to get(...)
        */
        template<protix_t U, typename... Ts>
        void set(Ts&... args) {
            return io<U>::set(*this, args...);
        }

      private:
        static constexpr unsigned short RK = bpprix::decode<bpprix::kR>(T) * bpprix::decode<bpprix::kK>(T) 
                                                                           + bpprix::decode<bpprix::kI>(T) ; // Max number of bases over all ridge functions

        template<protix_t U> requires proto::file_t<U>
        const char(&suffix())[std::size(bppr::SUFFIX)] { return bppr::SUFFIX; }
        template<> const char(&suffix<proto::kFUpdate>())[std::size(bppr::SUFFIX)] { return bppr::UPDTF; }

        template<protix_t U> requires proto::file_t<U>
        const auto mode() { return std::ios::out | std::ios::binary; }
        template<> const auto mode<proto::kFUpdate>() { return std::ios::in | std::ios::binary; }

        template<protix_t U> struct io;

        template<protix_t U> requires proto::nR_t<U> struct io<U> {
            static MKL_UINT offset() { return U*sizeof(proto_t); }
            static bool get(Proto<T>& p, unsigned short& nRidge) {
                proto_t nR;
                p._from.seekg(offset(), std::ios::beg);
                p._from.read(reinterpret_cast<char*>(&nR), sizeof(nR));
                nRidge = static_cast<unsigned short>(nR);
                return nRidge > 0;
            }
            static void set(Proto<T>& p, const proto_t nRidge) {
                p._from.seekp(offset(), std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(&nRidge), sizeof(nRidge));
            }
        };

        template<> struct io<proto::kY> {
            static MKL_UINT offset() { return proto::kN*sizeof(proto_t); }
            static bool get(Proto<T>& p, float(&y)[bpprix::decode<bpprix::kN>(T)]) {
                proto_t yOffset;
                p._from.seekg(proto::kY*sizeof(proto_t), std::ios::beg);
                p._from.read(reinterpret_cast<char*>(&yOffset), sizeof(yOffset));
                if (yOffset == kAbsent)  { return false; }
                p._from.seekg(yOffset , std::ios::beg);
                p._from.read(reinterpret_cast<char*>(y), sizeof(y));
                return true;
            }
            static void set(Proto<T>& p, const float(&y)[bpprix::decode<bpprix::kN>(T)]) {
                const proto_t yOffset = offset();
                p._from.seekp(proto::kY*sizeof(proto_t), std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(&yOffset), sizeof(yOffset));
                p._from.seekp(yOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(y), sizeof(y));
            }
        };

        template<> struct io<proto::kProj> {
            static MKL_UINT offset() { return io<proto::kY>::offset() + bpprix::decode<bpprix::kN>(T)*sizeof(float); }
            static bool get(Proto<T>& p, float(&X)[bpprix::decode<bpprix::kN>(T) * bpprix::decode<bpprix::kM>(T)]
                                      , float(&transform)[bpprix::decode<bpprix::kM>(T) * bpprix::decode<bpprix::kR>(T)]
                                      , float(&transforms)[bpprix::decode<bpprix::kP>(T)*bpprix::decode<bpprix::kM>(T)*bpprix::decode<bpprix::kR>(T)]) {
                unsigned short nRidge;
                if (!io<proto::kNRidge>::get(p, nRidge)) { return false; }
                proto_t projOffset;
                p._from.seekg(proto::kProj*sizeof(proto_t), std::ios::beg);
                p._from.read(reinterpret_cast<char*>(&projOffset), sizeof(projOffset));
                if (projOffset == kAbsent)  { return false; }
                p._from.seekg(projOffset, std::ios::beg);
                p._from.read(reinterpret_cast<char*>(X), sizeof(X));
                p._from.read(reinterpret_cast<char*>(transform), bpprix::decode<bpprix::kM>(T)*nRidge*sizeof(float));

                projOffset = io<proto::kCache>::template offset<proto::kProj>(nRidge);
                if (io<proto::kRC>::get(p, nRidge)) {
                    p._from.seekg(projOffset, std::ios::beg);
                    p._from.read(reinterpret_cast<char*>(transforms), nRidge*bpprix::decode<bpprix::kM>(T)*sizeof(float));
                }
                return true;
            }

            static void set(Proto<T>& p, const unsigned short nRidge
                                     , const float(&X)[bpprix::decode<bpprix::kN>(T) * bpprix::decode<bpprix::kM>(T)]
                                     , const float(&transform)[bpprix::decode<bpprix::kM>(T) * bpprix::decode<bpprix::kR>(T)]
                                     , const unsigned short nRC
                                     , const float(&transforms)[bpprix::decode<bpprix::kP>(T)*bpprix::decode<bpprix::kM>(T)*bpprix::decode<bpprix::kR>(T)]) {
                proto_t projOffset = offset();
                p._from.seekp(proto::kProj*sizeof(proto_t), std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(&projOffset), sizeof(projOffset));
                p._from.seekp(projOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(X), sizeof(X));
                p._from.write(reinterpret_cast<const char*>(transform), bpprix::decode<bpprix::kM>(T)*nRidge*sizeof(float));

                projOffset = io<proto::kCache>::template offset<proto::kProj>(nRidge);
                p._from.seekp(projOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(transforms), nRC*bpprix::decode<bpprix::kM>(T)*sizeof(float));
            }
        };

        template<> struct io<proto::kKnot> {
            static MKL_UINT offset(const unsigned short nRidge) { 
                return io<proto::kProj>::offset() + bpprix::decode<bpprix::kM>(T)*(bpprix::decode<bpprix::kN>(T) + nRidge)*sizeof(float);
            }
            static bool get(Proto<T>& p, unsigned short& nRidge, float(&nu)[bpprix::decode<bpprix::kR>(T)]
                                      , float(&knots)[bpprix::decode<bpprix::kP>(T)*bpprix::decode<bpprix::kR>(T)*(bpprix::decode<bpprix::kK>(T)+2)]) {
                if (!io<proto::kNRidge>::get(p, nRidge)) { return false; }
                proto_t knotOffset;
                p._from.seekg(proto::kKnot*sizeof(proto_t), std::ios::beg);
                p._from.read(reinterpret_cast<char*>(&knotOffset), sizeof(knotOffset));
                if (knotOffset == kAbsent)  { return false; }
                p._from.seekg(knotOffset, std::ios::beg);
                p._from.read(reinterpret_cast<char*>(nu), nRidge*sizeof(float));

                unsigned short nRC = 0;
                if (io<proto::kRC>::get(p, nRC)) {
                    knotOffset = io<proto::kCache>::template offset<proto::kKnot>(nRidge, nRC);
                    p._from.seekg(knotOffset, std::ios::beg);
                    p._from.read(reinterpret_cast<char*>(knots), nRC*(bpprix::decode<bpprix::kK>(T)+2)*sizeof(float));
                }
                return true;
            }
            static void set(Proto<T>& p, const unsigned short nRidge, const float(&nu)[bpprix::decode<bpprix::kR>(T)]
                                     , const unsigned short nRC, const float(&knots)[bpprix::decode<bpprix::kP>(T)*bpprix::decode<bpprix::kR>(T)*(bpprix::decode<bpprix::kK>(T)+2)]) {
                proto_t knotOffset = offset(nRidge);
                p._from.seekp(proto::kKnot*sizeof(proto_t), std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(&knotOffset), sizeof(knotOffset));
                p._from.seekp(knotOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(nu), nRidge*sizeof(float));

                knotOffset = io<proto::kCache>::template offset<proto::kKnot>(nRidge, nRC);
                p._from.seekp(knotOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(knots), nRC*(bpprix::decode<bpprix::kK>(T)+2)*sizeof(float));
            }
        };

        template<> struct io<proto::kZS> {
            static MKL_UINT offset(const unsigned short nRidge) { return io<proto::kKnot>::offset(nRidge) + nRidge*sizeof(float); }
            static bool get(Proto<T>& p, unsigned short& nRidge
                                      , float(&y)[bpprix::decode<bpprix::kN>(T)]
                                      , float& tau
                                      , float(&indexW)[bpprix::decode<bpprix::kM>(T)+2]
                                      , float(&activeW)[bpprix::decode<bpprix::kA>(T)+2]
                                      , unsigned short(&nRs)[bpprix::decode<bpprix::kP>(T)]
                                      , float(&betas)[RK*bpprix::decode<bpprix::kP>(T)]) {
                if (!io<proto::kNRidge>::get(p, nRidge)) { return false; }
                if (!io<proto::kY>::get(p, y)) { return false; }
                proto_t zsOffset;
                p._from.seekg(proto::kZS*sizeof(proto_t), std::ios::beg);
                p._from.read(reinterpret_cast<char*>(&zsOffset), sizeof(zsOffset));
                if (zsOffset == kAbsent)  { return false; }
                p._from.seekg(zsOffset, std::ios::beg);
                p._from.read(reinterpret_cast<char*>(&tau), sizeof(float));
                p._from.read(reinterpret_cast<char*>(indexW), (bpprix::decode<bpprix::kM>(T)+2)*sizeof(float));
                p._from.read(reinterpret_cast<char*>(activeW), (bpprix::decode<bpprix::kA>(T)+2)*sizeof(float));

                unsigned short nRC = 0;
                if (io<proto::kRC>::get(p, nRC)) {
                    zsOffset = io<proto::kCache>::template offset<proto::kZS>(nRidge, nRC);
                    nRC = nRC*bpprix::decode<bpprix::kK>(T) + bpprix::decode<bpprix::kI>(T)*bpprix::decode<bpprix::kP>(T);
                    p._from.seekg(zsOffset, std::ios::beg);
                    p._from.read(reinterpret_cast<char*>(nRs), bpprix::decode<bpprix::kP>(T)*sizeof(short));
                    p._from.read(reinterpret_cast<char*>(betas), nRC*sizeof(float));
                }
                return true;
            }
            static void set(Proto<T>& p, const unsigned short nRidge
                                      , const float(&y)[bpprix::decode<bpprix::kN>(T)]
                                      , const float& tau
                                      , const float(&indexW)[bpprix::decode<bpprix::kM>(T)+2]
                                      , const float(&activeW)[bpprix::decode<bpprix::kA>(T)+2]
                                      , const unsigned short& nRC
                                      , const unsigned short(&nRs)[bpprix::decode<bpprix::kP>(T)]
                                      , const float(&betas)[RK*bpprix::decode<bpprix::kP>(T)]) {
                io<proto::kNRidge>::set(p, nRidge);
                io<proto::kY>::set(p, y);
                proto_t zsOffset = offset(nRidge);
                p._from.seekp(proto::kZS*sizeof(proto_t), std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(&zsOffset), sizeof(zsOffset));
                p._from.seekp(zsOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(&tau), sizeof(float));
                p._from.write(reinterpret_cast<const char*>(indexW), (bpprix::decode<bpprix::kM>(T)+2)*sizeof(float));
                p._from.write(reinterpret_cast<const char*>(activeW), (bpprix::decode<bpprix::kA>(T)+2)*sizeof(float));

                io<proto::kRC>::set(p, nRC);
                zsOffset = io<proto::kCache>::template offset<proto::kZS>(nRidge, nRC);
                p._from.seekp(zsOffset, std::ios::beg);
                p._from.write(reinterpret_cast<const char*>(nRs), bpprix::decode<bpprix::kP>(T)*sizeof(short));
                zsOffset =  nRC*bpprix::decode<bpprix::kK>(T) + bpprix::decode<bpprix::kI>(T)*bpprix::decode<bpprix::kP>(T);
                p._from.write(reinterpret_cast<const char*>(betas), zsOffset*sizeof(float));
            }
        };

        template<> struct io<proto::kCache> {
            template<protix_t U, typename... Ts>
            static MKL_UINT offset(Ts... args) { return proto::kCache*sizeof(proto_t); }
            template<> MKL_UINT offset<proto::kProj>(const unsigned short nRidge) { 
                return io<proto::kZS>::offset(nRidge) + (bpprix::decode<bpprix::kM>(T) + bpprix::decode<bpprix::kA>(T) + 5)*sizeof(float); 
            }
            template<> MKL_UINT offset<proto::kKnot>(const unsigned short nRidge, const unsigned short nRC) { 
                return offset<proto::kProj>(nRidge) + bpprix::decode<bpprix::kM>(T)*nRC*sizeof(float); 
            }
            template<> MKL_UINT offset<proto::kZS>(const unsigned short nRidge, const unsigned short nRC) { 
                return offset<proto::kKnot>(nRidge, nRC) + (bpprix::decode<bpprix::kK>(T)+2)*nRC*sizeof(float); 
            }
            template<> MKL_UINT offset<proto::kCache>(const unsigned short nRidge, const unsigned short nRC) { 
                return offset<proto::kZS>(nRidge, nRC) + (nRC*bpprix::decode<bpprix::kK>(T)
                            + bpprix::decode<bpprix::kP>(T)*(1+bpprix::decode<bpprix::kI>(T)))*sizeof(float); 
            }
            static bool get(Proto<T>& p, MKL_UINT(&rcx)[rcx::N]) {
                unsigned short nRidge, nRC;
                if (!io<proto::kNRidge>::get(p, nRidge)) { return false; }
                if (!io<proto::kRC>::get(p, nRC)) { return false; }
                const proto_t cOffset =  offset<proto::kCache>(nRidge, nRC);
                p._from.seekg(cOffset, std::ios::beg);
                p._from.read(reinterpret_cast<char*>(rcx), rcx::N*sizeof(MKL_UINT));
                return true;
            }
            static void set(Proto<T>& p, const MKL_UINT(&rcx)[rcx::N]) {
                unsigned short nRidge;
                if (io<proto::kNRidge>::get(p, nRidge)) {
                    const proto_t cOffset = offset<proto::kCache>(nRidge, rcx[rcx::kRx]);
                    p._from.seekp(cOffset, std::ios::beg);
                    p._from.write(reinterpret_cast<const char*>(rcx), rcx::N*sizeof(MKL_UINT));
                }
            }
        };

        template<> struct io<proto::kFUpdate> {
            static bool get(Proto<T>& p, float(&X)[bpprix::decode<bpprix::kM>(T)], float& y) {
                p._from.read(reinterpret_cast<char*>(&y), sizeof(float));
                p._from.read(reinterpret_cast<char*>(X), bpprix::decode<bpprix::kM>(T)*sizeof(float));
                if (p._from) { return true; }
                return false;
            }
        };

        std::fstream _from;
    };

} // namespace bppr