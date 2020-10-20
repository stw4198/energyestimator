import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import ROOT as root

f_water = root.TFile("../../rat-pac/output/water/testing/b_monopositrons.root", "READ")
f_wbls1pct = root.TFile("../../rat-pac/output/wbls1pct/testing/b_monopositrons.root", "READ")
f_wbls3pct = root.TFile("../../rat-pac/output/wbls3pct/testing/bonsai_monopositrons_wbls3.root", "READ")
f_wbls5pct = root.TFile("../../rat-pac/output/wbls5pct/testing/b_monopositrons.root", "READ")

# water fits
fits_water = {
    "water n9": ("0.142414*n9py+1.3434", "n9py"),
    "water n9eff": ("0.0332394*n9eff+1.64611", "n9eff")
}

fits_wbls1pct = {
    "wbls1% n100": ("0.0394906*n100py-0.290672", "n100py"),
    "wbls1% n100eff": ("0.00880797*n100eff+0.138467", "n100eff")
}

fits_wbls3pct = {
    "wbls3% n100": ("0.0290602*n100py-0.673947", "n100py"),
    "wbls3% n100eff": ("0.00612595*n100eff-0.177031", "n100eff")
}

fits_wbls5pct = {
    "wbls5% n100": ("0.0232902*n100py-0.888654", "n100py"),
    "wbls5% n100eff": ("0.00475615*n100eff-0.378911", "n100eff")
}

files = {
    #f_water: fits_water,
    #f_wbls1pct: fits_wbls1pct,
    #f_wbls3pct: fits_wbls3pct,
    f_wbls5pct: fits_wbls5pct
}

hist_idx = 0
f1, a1 = plt.subplots()
f2, a2 = plt.subplots()

for file in files:

    #filename = sys.argv[1]
    #rfile = root.TFile(filename, "READ")
    tree = file.Get("data")
    energies = np.arange(0.5, 10.5, 0.5)
    #fig, axs = plt.subplots(2)
    fits = files[file]
    #hist_idx = 0
    for key in fits:
        #hist_idx += 1
        res = []
        mean = []
        res_error = []
        mean_error = []
        xs = []
        for e in energies:
            tree.Draw(f"({fits[key][0]})-mc_energy>>hist{str(hist_idx)}", f"{fits[key][1]}>0 && closestPMT>0 && mc_energy=={e}", "goff")
            hist = root.gDirectory.Get(f"hist{str(hist_idx)}")
            hist_idx += 1
            if hist.GetEntries() < 50: continue
            sigma = hist.GetStdDev()
            if sigma == 0: breakpoint()
            res.append(sigma / e)
            res_error.append(hist.GetStdDevError() / e)
            tree.Draw(f"({fits[key][0]})>>hist1{str(hist_idx)}", f"{fits[key][1]}>0 && closestPMT>0 && mc_energy=={e}", "goff")
            hist1 = root.gDirectory.Get(f"hist1{str(hist_idx)}")
            hist_idx += 1
            vmean = hist1.GetMean()
            mean.append(vmean / e)
            mean_error.append(hist1.GetMeanError() / e)
            xs.append(e)
            #hist.Reset("ICESM")
        a1.errorbar(xs, res, yerr=res_error, label=key)
        a2.errorbar(xs, mean, yerr=mean_error, label=key)

a1.set_xticks(np.arange(0.5, 10.5, 0.5))
a1.set_ylabel("Sigma / E")
a1.set_xlabel("MC energy")
a2.set_xticks(np.arange(0.5, 10.5, 0.5))
a2.set_ylabel("Mean reco energy / MC energy")
a2.set_xlabel("MC energy")
plt.close(f1)
plt.legend()
plt.show()
print("Finished")
