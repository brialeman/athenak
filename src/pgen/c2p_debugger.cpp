//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file c2p_debugger.cpp
//  \brief Single-point tester for C2P

#include <stdio.h>
#include <math.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

template<class EOSPolicy>
void RunTest(ParameterInput *pin, Mesh* pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // Horrible cast so we can get access to the EOS and PrimitiveSolver
  dyngr::DynGRMHDPS<EOSPolicy, Primitive::ResetFloor>* pdyngrps =
    static_cast<dyngr::DynGRMHDPS<EOSPolicy, Primitive::ResetFloor>*>(
      pmbp->pdyngr);
  auto& ps = pdyngrps->eos.ps;

  int& nscal_ = pmbp->pmhd->nscalars;

  Real cons_pt[NCONS], cons_pt_old[NCONS];
  cons_pt[CDN] = cons_pt_old[CDN] = pin->GetReal("problem", "D");
  cons_pt[CSX] = cons_pt_old[CSX] = pin->GetReal("problem", "Sx");
  cons_pt[CSY] = cons_pt_old[CSY] = pin->GetReal("problem", "Sy");
  cons_pt[CSZ] = cons_pt_old[CSZ] = pin->GetReal("problem", "Sz");
  cons_pt[CTA] = cons_pt_old[CTA] = pin->GetReal("problem", "tau");

  for (int s = 0; s < nscal_; s++) {
    std::stringstream ss;
    ss << "DY" << s;
    cons_pt[CYD + s] = cons_pt_old[CYD + s] = pin->GetReal("problem", ss.str());
  }

  Real Bu[NMAG];
  Bu[IBX] = pin->GetReal("problem", "Bx");
  Bu[IBY] = pin->GetReal("problem", "By");
  Bu[IBZ] = pin->GetReal("problem", "Bz");

  Real g3d[NSPMETRIC];
  g3d[S11] = pin->GetReal("problem", "gxx");
  g3d[S12] = pin->GetReal("problem", "gxy");
  g3d[S13] = pin->GetReal("problem", "gxz");
  g3d[S22] = pin->GetReal("problem", "gyy");
  g3d[S23] = pin->GetReal("problem", "gyz");
  g3d[S33] = pin->GetReal("problem", "gzz");

  Real detg = Primitive::GetDeterminant(g3d);

  Real g3u[NSPMETRIC];

  Primitive::InvertMatrix(g3u, g3d, detg);

  Real prim_pt[NPRIM];

  Primitive::SolverResult result = ps.ConToPrim(prim_pt, cons_pt, Bu, g3d, g3u);

  if (result.error != Primitive::Error::SUCCESS) {
    Kokkos::printf("The C2P failed!\n"
                   "  cons floor: %d\n"
                   "  prim floor: %d\n"
                   "  cons adjusted: %d\n",
                   result.cons_floor, result.prim_floor, result.cons_adjusted);
  } else {
    Kokkos::printf("The C2P succeeded!\n"
                   "  cons floor: %d\n"
                   "  prim floor: %d\n"
                   "  cons adjusted: %d\n",
                   result.cons_floor, result.prim_floor, result.cons_adjusted);
  }

  return;
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto& eos_policy = pmbp->pdyngr->eos_policy;
  if (eos_policy == DynGRMHD_EOS::eos_ideal) {
    RunTest<Primitive::IdealGas>(pin, pmy_mesh_);
  } else if (eos_policy == DynGRMHD_EOS::eos_compose) {
    bool use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT", false);
    if (use_NQT) {
      RunTest<Primitive::EOSCompOSE<Primitive::NQTLogs>>(pin, pmy_mesh_);
    } else {
      RunTest<Primitive::EOSCompOSE<Primitive::NormalLogs>>(pin, pmy_mesh_);
    }
  } else if (eos_policy == DynGRMHD_EOS::eos_hybrid) {
    bool use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT", false);
    if (use_NQT) {
      RunTest<Primitive::EOSHybrid<Primitive::NQTLogs>>(pin, pmy_mesh_);
    } else {
      RunTest<Primitive::EOSHybrid<Primitive::NormalLogs>>(pin, pmy_mesh_);
    }
  } else if (eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    RunTest<Primitive::PiecewisePolytrope>(pin, pmy_mesh_);
  }

  pmbp->padm->SetADMVariables(pmbp);
}
