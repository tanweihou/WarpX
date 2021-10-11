/* Copyright 2021 Wei Hou Tan
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MACROSCOPICPROPERTIESCONDUCTIVITYBOOSTED_H_
#define WARPX_MACROSCOPICPROPERTIESCONDUCTIVITYBOOSTED_H_

#include "MacroscopicProperties_fwd.H"

#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>

/**
 * \brief
 * This struct contains only static functions to compute the co-efficients for the
 * Lax-Wendroff scheme of macroscopic Maxwells equations using
 * macroscopic properties, namely, conductivity (sigma), permittivity (epsilon).
 * This struct takes into account of moving conductors, where we need to consider
 * the covariant generalization of Ohm's law.
 */
struct LaxWendroffAlgoBoostConductor {
     
     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha1xy (amrex::Real const sigma,
                                  amrex::Real const epsilon,
                                  amrex::Real dt,
                                  amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * sigma * dt * gamma_boost / epsilon;
         amrex::Real alpha = (1._rt - fac1) / (1._rt + fac1);
         return alpha;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha2xy (amrex::Real const sigma,
                                  amrex::Real const epsilon,
                                  amrex::Real dt,
                                  amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * sigma * dt * gamma_boost / epsilon;
         amrex::Real alpha = dt / (epsilon * (1._rt + fac1));
         return alpha
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha3xy (amrex::Real const sigma,
                                  amrex::Real const epsilon,
                                  amrex::Real dt,
                                  amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real clight = PhysConst::c;
         amrex::Real fac1 = 0.5_rt * sigma * dt * gamma_boost / epsilon;
         amrex::Real beta_boost = std::sqrt( 1._rt - 1._rt/( gamma_boost * gamma_boost) );
         amrex::Real alpha = beta_boost * clight * fac1 / (1._rt + fac1);
         return alpha;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha1z (amrex::Real const sigma,
                                 amrex::Real const epsilon,
                                 amrex::Real dt,
                                 amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * sigma * dt / (epsilon * gamma_boost);
         amrex::Real alpha1 = (1._rt - fac1) / (1._rt + fac1);
         return alpha1;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha2z (amrex::Real const sigma,
                                 amrex::Real const epsilon,
                                 amrex::Real dt,
                                 amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = 0.5_rt * sigma * dt / (epsilon * gamma_boost);
         amrex::Real alpha = dt / (epsilon * (1._rt + fac1));
         return alpha
     }

}

/**
 * \brief
 * This struct contains only static functions to compute the co-efficients for the
 * BackwardEuler scheme of macroscopic Maxwells equations using
 * macroscopic properties, namely, conductivity (sigma) and permittivity (epsilon).
 * This struct takes into account of moving conductors, where we need to consider
 * the covariant generalization of Ohm's law.
 */
struct BackwardEulerAlgoBoostConductor {

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha1xy (amrex::Real const sigma,
                                  amrex::Real const epsilon,
                                  amrex::Real dt,
                                  amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = sigma * dt * gamma_boost / epsilon;
         amrex::Real alpha = (1._rt) / (1._rt + fac1);
         return alpha;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha2xy (amrex::Real const sigma,
                                  amrex::Real const epsilon,
                                  amrex::Real dt,
                                  amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = sigma * dt * gamma_boost / epsilon;
         amrex::Real alpha = dt / (epsilon * (1._rt + fac1));
         return alpha
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha3xy (amrex::Real const sigma,
                                  amrex::Real const epsilon,
                                  amrex::Real dt,
                                  amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real clight = PhysConst::c;
         amrex::Real fac1 = sigma * dt * gamma_boost / epsilon;
         amrex::Real beta_boost = std::sqrt( 1._rt - 1._rt/( gamma_boost * gamma_boost) );
         amrex::Real alpha = 0.5_rt * beta_boost * clight * fac1 / (1._rt + fac1);
         return alpha;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha1z (amrex::Real const sigma,
                                 amrex::Real const epsilon,
                                 amrex::Real dt,
                                 amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = sigma * dt / (epsilon * gamma_boost);
         amrex::Real alpha1 = (1._rt) / (1._rt + fac1);
         return alpha1;
     }

     AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
     static amrex::Real alpha2z (amrex::Real const sigma,
                                 amrex::Real const epsilon,
                                 amrex::Real dt,
                                 amrex::Real gamma_boost) {
         using namespace amrex;
         amrex::Real fac1 = sigma * dt / (epsilon * gamma_boost);
         amrex::Real alpha = dt / (epsilon * (1._rt + fac1));
         return alpha
     }

};