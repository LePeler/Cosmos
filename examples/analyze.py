import os
import sys
import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer

folder = os.path.split(__file__)[0]

if len(sys.argv) == 1:
    raise ValueError("Please specify the model which you want to analyze as a commandline argument.")
else:
    model = sys.argv[1]

if (model not in ["LCDM", "PowerLaw", "quintessence"]):
    raise ValueError(f"Please specify one of the following models: 'LCDM', 'PowerLaw' or 'quintessence', not {model}.")

columns = {
    "LCDM": ["H0 [km/s/Mpc]", "Om0", "M"],
    "PowerLaw": ["H0 [km/s/Mpc]", "Om0", "b", "M"],
    "quintessence": ["H0 [km/s/Mpc]", "Om0", "T0", "l/H0", "M"],
}
title = {
    "LCDM": "LCDM (CC+SN1a+BAO)",
    "PowerLaw": "f(T) power law (CC+SN1a)",
    "quintessence": "exponential quintessence (CC+SN1a)",
}

N_eff = 5000

c = ChainConsumer()

data = np.genfromtxt(f"{folder}/{model}_output.txt", delimiter=",")
states = data[:,:-1]
logprobs = data[:,-1]
states_df = pd.DataFrame(states, columns=columns[model])
logL_max = np.max(logprobs)
num_params = states.shape[1]
AIC = 2*num_params -2*logL_max
BIC = num_params*np.log(N_eff) -2*logL_max

c.add_chain(Chain(samples=states_df, name=model))

print("ln(P_max) =", logL_max)
print("AIC =", AIC)
print("BIC =", BIC)


summ = c.analysis.get_summary()[model]
for k, v in summ.items():
    center = v.center
    low = v.center-v.lower if v.lower is not None else np.nan
    high = v.upper-v.center if v.upper is not None else np.nan
    print(f"{k} = {center} + {high} - {low}")


c.plotter.config.dpi = 1000


fig = c.plotter.plot(figsize="column")

fig.suptitle(title[model], fontsize="32")

fig.savefig(f"{folder}/{model}_plot.pdf")

