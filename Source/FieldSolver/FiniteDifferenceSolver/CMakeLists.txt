target_sources(WarpX
  PRIVATE
    ComputeDivE.cpp
    EvolveB.cpp
    EvolveBPML.cpp
    EvolveE.cpp
    EvolveEPML.cpp
    EvolveF.cpp
    EvolveFPML.cpp
    EvolveG.cpp
    EvolveECTRho.cpp
    FiniteDifferenceSolver.cpp
    MacroscopicEvolveE.cpp
    ApplySilverMuellerBoundary.cpp
    MacroscopicEvolveEBoostConductor.cpp
)

add_subdirectory(MacroscopicProperties)
