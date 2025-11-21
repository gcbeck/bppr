/**
 * *****************************************************************************
 * \file bppr.cpp
 * \author Graham Beck
 * \brief BPPR: Bayesian Projection Pursuit Regression training & demonstration executable.
 *                        The algorithm is based on the work by Collins et al 
 *                        at https://github.com/gqcollins/BayesPPR.
 * \version 0.1
 * \date 2025-11-21
 *
 * \copyright Copyright (c) 2025
 * *****************************************************************************
 */
#include "bppr.h"

#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <sstream> 
#include <string>
#include <unistd.h>

#define MKL_DIRECT_CALL

#include "mkl_service.h"
#include "mkl_vml_defines.h"
#include "mkl_vml_functions.h"

constexpr auto REPO = join("..", bppr::PATHSEP, "dat", bppr::PATHSEP);
constexpr auto T = bppr::bpprix::encode(256, 3, 8, 2, 4, 16, false);

constexpr char OPTS[] = "e:t:d:w";
constexpr char OPTSEP = ',';

template <typename U>
concept Numeric = std::is_arithmetic_v<U>;

template<Numeric U>
U next(std::stringstream& ss) {
    std::string token;
    if (std::getline(ss, token, OPTSEP)) { 
        if constexpr (std::is_integral<U>::value) { return std::stoi(token); } else { return std::stof(token); }
    } else { 
        throw std::runtime_error("Unparseable Parameter"); 
    }
}

/**
* @brief Entry point of the application.
*
* @details Processes command-line arguments to:
*                 1) Relay back the BPPR encoding integer given the const BPPR parameters
* @example `./bppr -e 256,3,8,2,4,16,0
* @details 2) Initialize a BPPR object which includes the training 
* @example `./bppr -t 2 -d 256,16,4 -w
*
* @note The primary use case is as a library, where .update(X,y) and .predict<.>(X) can be 
*              called at will, hence the lack of updating/prediction options in the executable arguments. 
*              That functionality is however demonstrated below where it is assumed a '.updt' file 
*              with OOS rows exists, from which online updates/predictions are performed. 
*
* @param e The flag requesting an encoding of the argument N,M,R,A,K,P,I where:
*              N is the dataset length (number of rows)
*              M is the number of signals present in the dataset (number of cols)
*              R is the maximum number of ridge functions
*              A is the maximum number of interacting terms per ridge function
*              K is the number of knots, or degree of localization of the learning
*              P is the number of posterior terms defining the prediction distribution 
*              I is the boolean flag for whether an intercept term should be included
* @param t The number of threads to instruct Intel MKL to use
* @param d The flag listing the descriptor parameters nBurn,nAdapt,nEvery where:
*              nAdapt is the number of inital mcmc iterations to make to perform signal selection without shrinkage calculation
*              nBurn is the number of mcmc iterations to make that also perform shrinkage calculation but before posterior caching
*              nEvery is the period of posterior caching: each nEvery sample of the mcmc'd beta coefs are kept
* @param w The flag indicating that the trained BPPR should be written to the repo, overwriting any of the same encoding
*/
int main(int argc, char *argv[])
{
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_DEFAULT);  
    // If supremely confident in stability, consider VML_ERRMODE_DEFAULT->VML_ERRMODE_ERRNO

    int opt;
    bool write = false;
    std::stringstream dss;

    while ((opt = getopt(argc, argv, OPTS)) != -1) {
        switch (opt) {
          case 'e': {
            std::stringstream ss(optarg);
            std::cout << "Encoding: " << bppr::bpprix::encode(next<MKL_UINT>(ss), 
                                                                                                     next<MKL_UINT>(ss), 
                                                                                                     next<unsigned short>(ss), 
                                                                                                     next<unsigned short>(ss), 
                                                                                                     next<unsigned short>(ss), 
                                                                                                     next<MKL_UINT>(ss), 
                                                                                                     next<bool>(ss)) << std::endl;
            return 0;
          }
          case 't':
            mkl_set_num_threads(std::stoi(optarg));
            break;
          case 'd':
            dss << optarg;
            break;
          case 'w':
            write = true;
            break;
        }
    }

    // Construct the Descriptor containing the non-compile-time parameters. 
    const bppr::Descriptor<T> dscr(next<const MKL_UINT>(dss), next<const unsigned short>(dss), next<const unsigned short>(dss));

    // Initialize the L3 logging
    l3_init(join<T>(REPO, bppr::L3FILE).data());

    // Contruct the BPPR itself and perform the training
    bppr::BPPR<T> bppr(dscr, REPO);

    // Make all online predictions from the .updt file. Alternatively, bppr.update(REPO) to make online updates. 
    bppr.predict<bppr::pred::kDistribution>(REPO);

    // Persist the BPPR state to file if desired. 
    if (write) { bppr.write(REPO); }

    // Deinitialize the logging. 
    l3_deinit();

    return 0;
}