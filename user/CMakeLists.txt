
# test executable
add_cosmos_executable(cosmos_test
    test.cxx
)
# LCDM MCMC executable
add_cosmos_executable(run_LCDM
    mcmc/LCDM.cxx
)
# f1 MCMC executable
add_cosmos_executable(run_f1
    mcmc/f1.cxx
)
# f2 MCMC executable
add_cosmos_executable(run_f2
    mcmc/f2.cxx
)
# f3 MCMC executable
add_cosmos_executable(run_f3
    mcmc/f3.cxx
)
# f4 MCMC executable
add_cosmos_executable(run_f4
    mcmc/f4.cxx
)
# f5 MCMC executable
add_cosmos_executable(run_f5
    mcmc/f5.cxx
)

# SFT MCMC executable
add_cosmos_executable(run_SFT
    mcmc/SFT.cxx
)

