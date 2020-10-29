import argparse
import os
import pdb
import sys

import numpy as np
from scipy.interpolate import griddata, interp1d
from tqdm import tqdm

try:
    import ROOT as root
except ImportError:
    print("Could not import ROOT. Source thisroot.sh first.")
    exit(1)

# TODO: Option to create fit to energy and then calculate best guess
# TODO: Option to write best guess energy to tree

np.set_printoptions(suppress=True)


class Estimator():

    def __init__(self, rat_fn=None, bonsai_fn=None, min_t=-3, max_t=6,
                 dark_noise=10000., qe=0.3, pmt_sep=750., medium=None,
                 geometry=None, fitter=None, load_lib=True):
        self.rat_fn = rat_fn
        self.bonsai_fn = bonsai_fn
        self.min_t = min_t
        self.max_t = max_t
        self.window = self.max_t - self.min_t
        self.medium = medium
        self.geometry = geometry
        self.fitter = fitter
        self.media_optics = {"water": 117509,
                             "wbls1pct": 145312,
                             "wbls3pct": 135169,
                             "wbls5pct": 126889}
        self.pmt_sep = pmt_sep
        self.dark_noise = dark_noise
        self.qe = qe
        self.load_lib = load_lib

    def parse_options(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("rat_fn", help="filepath to the rat-pac file",
                            type=str)
        parser.add_argument("bonsai_fn", help="filepath to the bonsai file \
                            containing reconstructed events", type=str)
        parser.add_argument("--minT", help="start time of rolling window \
                            relative to current PMT hit time in ns, typically \
                            a negative number or zero \
                            (default: -3)", type=int, default=-3)
        parser.add_argument("--maxT", help="end time of rolling window in ns \
                            (default: 6)",
                            type=int, default=6)
        parser.add_argument("--medium", help="specify the detection medium \
                            (default: water)",
                            type=str, choices=['water', 'wbls1pct',
                                               'wbls3pct', 'wbls5pct'],
                            default='water')
        parser.add_argument("--darkNoise", help="specify the PMT dark noise \
                            (default: 3000)",
                            type=float, default=3000.)
        parser.add_argument("--qe", help="Specify the PMT quantum efficiency \
                            (default: 0.3)",
                            type=float, default=0.3)
        parser.add_argument("--pmtSep", help="specify the average PMT \
                            diagonal separation, set to 0 to calculate dynamically \
                            (default: 750)", type=float, default=750.)
        parser.add_argument("--fitter", help="specify which fitter to analyse: \
                            bonsai or qfit \
                            (default: bonsai)", type=str, default='bonsai')
        args = parser.parse_args(argv)
        if self.load_lib:
            root.gSystem.Load('libRATEvent.so')
            print("ROOT: Loaded libRATEvent.so")
        self.rat_fn = args.rat_fn
        self.rat_file = self.is_valid_file(parser, self.rat_fn)
        self.bonsai_fn = args.bonsai_fn
        self.bonsai_file = self.is_valid_file(parser, self.bonsai_fn, type="UPDATE")
        self.min_t = args.minT
        self.max_t = args.maxT
        self.window = self.max_t - self.min_t
        self.medium = args.medium
        self.dark_noise = args.darkNoise
        self.qe = args.qe
        self.pmt_sep = args.pmtSep
        self.fitter = args.fitter
        self.transparency = self.media_optics[self.medium]
        self.get_file_data()
        if self.pmt_sep == 0: self.pmt_sep = self.calculate_pmt_separation()

    def get_file_data(self):
        self.rat_t = self.rat_file.Get("T")
        self.rat_rt = self.rat_file.Get("runT")
        self.bonsai_t = self.bonsai_file.Get("data")
        self.bonsai_rt = self.bonsai_file.Get("runSummary")
        try:
            self.rat_rt.GetEntry(0)
        except AttributeError:
            print(f"Error getting rat run tree from file {self.rat_fn}.")
            exit(1)
        self.pmtinfo = self.rat_rt.run.GetPMTInfo()

    def calculate_pmt_separation(self, buffer=50):
        print("Calculating PMT separation from PMTInfo")
        pmt_pos = []
        for pmt in range(self.pmtinfo.GetPMTCount()):
            # All PMTs are type 1 in the files that I have
            if self.pmtinfo.GetType(pmt) == 1:
                pmt_pos.append(np.array(self.pmtinfo.GetPosition(pmt)))
        pmt_pos = np.array(pmt_pos)
        radial_sep = np.max(np.diff(np.unique(np.sqrt(pmt_pos[:,0]**2 + pmt_pos[:,1]**2))))
        z_sep = np.max(np.diff(np.unique(pmt_pos[:,2])))
        pmt_separation = np.sqrt(radial_sep**2 + z_sep**2)
        print(f"Setting PMT separation to {pmt_separation + buffer}.")
        return pmt_separation + buffer

    def eventloop(self, write_tree=True):#, fitter='bonsai'):
        r_nentry = self.rat_t.GetEntries()
        b_nentry = self.bonsai_t.GetEntries()
        geo_data = self.parse_geometric_corrections("geo_correction.csv")
        final_row, max_height = calculate_final_row_height(self.pmtinfo)
        nXeff_values = []
        nX_values = []
        fitter = self.fitter
        print(f"There are {b_nentry} reconstructed events in {self.bonsai_fn}")
        for b_entry, b_event in enumerate(tqdm(self.bonsai_t, total=b_nentry)):
            self.rat_t.GetEntry(b_event.mcid)
            evcount = self.rat_t.ds.GetEVCount()
            if evcount == 0:
                print(f"No hits for this event, skipping.")
                continue
            for ev_idx in range(evcount):
                if ev_idx != b_event.subid:
                    continue
                assert b_event.mcx == self.rat_t.ds.GetMC().GetMCParticle(0).GetPosition().X()
                if fitter == 'qfit':
                    reco_vertex = np.array([b_event.xQFit, b_event.yQFit, b_event.zQFit])
                else:
                    reco_vertex = np.array([b_event.x, b_event.y, b_event.z])
                if np.any(reco_vertex < -9998):
                    nXeff_values.append(-9999.9)
                    nX_values.append(-9999.9)
                    continue
                hit_pmts = []
                ev = self.rat_t.ds.GetEV(ev_idx)
                hitpmtcount = ev.GetPMTCount()
                for pmt_idx in range(hitpmtcount):
                    pmt = ev.GetPMT(pmt_idx)
                    hit_pmts.append(pmt)
                corrected_hit_times = correct_tof(reco_vertex, self.pmtinfo, hit_pmts)
                nX, nXmask = nXhits(corrected_hit_times, self.min_t, self.max_t)
                nX_values.append(nX)
                n2X, _ = nXhits(corrected_hit_times, 2*self.min_t, 2*self.max_t)
                nXPMTs = np.array(hit_pmts)[nXmask]
                nXPMTpos = np.array([np.array(self.pmtinfo.GetPosition(pmt.GetID())) for pmt in nXPMTs])
                nXeff = 0.
                for hit in range(nX):
                    multihit_cor = new_occupancy(nXPMTpos, hit, self.pmt_sep, row_height=final_row, roof_height=max_height)
                    late_cor = tail_hits(nX, n2X, self.window, self.dark_noise)
                    dark_cor = dark_hits(nX, self.window, self.dark_noise)
                    #geo_cor = geometric1D(geo_data[:,0], geo_data[:,1], reco_vertex)
                    transp_cor = transparency(nXPMTpos[hit], reco_vertex, self.transparency)
                    qe_cor = quantum_efficiency(self.qe)
                    nXeff += (multihit_cor + late_cor - dark_cor) * transp_cor * qe_cor
                geo_cor = geometric1D(geo_data[:,0], geo_data[:,1], reco_vertex)
                nXeff *= geo_cor
                nXeff_values.append(nXeff)
        if write_tree:
            self.write_to_tree(nXeff_values, "n"+str((self.max_t-self.min_t))+"eff")
            self.write_to_tree(nX_values, "n"+str((self.max_t-self.min_t))+"py")
                
    @staticmethod
    def is_valid_file(parser, arg, type="READ"):
        if not os.path.exists(arg):
            parser.error(f"File {arg} does not exist.")
        else:
            return root.TFile(arg, type)

    def write_to_tree(self, values, br_name):
        # write values to bonsai tree under branch br_name
        vals_to_write = np.empty((1), dtype="double")
        branch = self.bonsai_t.Branch(br_name, vals_to_write, br_name+"/D")
        for value in values:
            vals_to_write[0] = value
            branch.Fill()
        self.bonsai_file.Write("", root.TFile.kOverwrite)

    def read_from_tree(self, br_name):
        # Read values of branch br_name from bonsai tree into np array
        values = self.bonsai_t.AsMatrix(columns=[br_name])
        return values

    @staticmethod
    def parse_geometric_corrections(filename):
        data = np.loadtxt(filename, skiprows=1, delimiter=",")
        return data

    def make_fit(self, estimator, conditions):
        self.bonsai_t.Draw("mc_energy:"+estimator+">>hist", conditions, "goff")
        hist = root.gDirectory.Get("hist")
        hist.Fit("pol1", "goff")
        p0 = hist.GetFunction("pol1").GetParameter(0)
        p1 = hist.GetFunction("pol1").GetParameter(1)
        fit = (p0, p1)
        return fit

    @staticmethod
    def estimate_energy(nXeff, fit):
        # pass individual value or array
        estimate = fit[0] + (fit[1] * nXeff)
        if isinstance(estimate, np.ndarray):
            estimate[estimate<0] = -9999.9
        return estimate


def correct_tof(vertex, pmtinfo, evPMTs, SoL=218):
    """Correct PMT hit times for photon time of flight.

    Args:
        vertex (array): The reconstructed event vertex (x,y,z)
        pmtinfo (RAT::DS::PMTInfo): RAT runTree PMTInfo object
        evPMTs (list[RAT::DS::PMT]): Collection of triggered PMTs

    Returns:
        hit_times_tof (array): Collection of tof-corrected PMT hit times
    """
    hit_times = np.array([pmt.GetTime() for pmt in evPMTs])
    hit_pos = np.array([np.array(pmtinfo.GetPosition(pmt.GetID())) for pmt in evPMTs])
    # Now calculate the time of flight...
    tofs = np.array([(fastmag(pathvec) / SoL) for pathvec in hit_pos - vertex])
    # ...and correct the hit times for time of flight
    hit_times_tof = hit_times - tofs
    return hit_times_tof

def nXhits(hit_times, minT, maxT):
    """Calculate max PMT hits in a time window of size maxT minus minT

    Args:
        hit_times (array): tof-corrected PMT hit times
        minT (float): Start time of rolling window relative to current PMT hit
                      time in ns, typically a negative number or zero
        maxT (float): End time of rolling window in ns

    Returns:
        nwindow (int): Maximum PMT hits in time window for the event
        filtered_hits (array): Boolean mask of PMT objects/hit times that pass
                               timing window filter.
    """
    nwindow = 0
    for time in hit_times:
        thisfilter = (hit_times>time+minT) & (hit_times<time+maxT)
        thiswindow = np.count_nonzero(thisfilter)
        if thiswindow > nwindow:
            filtered_hits = thisfilter
            nwindow = thiswindow
    return nwindow, filtered_hits

def fastmag(vector):
    # Faster magnitude calculation than linalg.norm or elementwise operation
    return np.sqrt(vector.dot(vector))

def xyz_to_rz(vertex):
    # For geometric correction
    assert len(vertex) == 3
    r = fastmag(vertex[:2])
    z = np.abs(vertex[2])
    return np.array([r, z])

def calculate_final_row_height(pmtinfo, buffer=50):
    pmt_locs = []
    for pmt in range(pmtinfo.GetPMTCount()):
        pmt_locs.append(pmtinfo.GetPosition(pmt))
    pmt_locs = np.array(pmt_locs)
    z_locs = np.unique(pmt_locs[:,2])
    final_row = min(abs(z_locs[1]), abs(z_locs[-2])) - buffer
    max_height = np.max(np.abs(z_locs)) - buffer
    return final_row, max_height

def occupancy(pmt_positions, ihit, nX, pmt_separation):
    nearby_hits = 0
    ipos = pmt_positions[ihit]
    for jhit in range(nX):
        if ihit != jhit:
            jpos = pmt_positions[jhit]
            ij_separation = fastmag(ipos - jpos)
            if ij_separation <= pmt_separation:
                nearby_hits += 1
    hit_ratio = nearby_hits / 8.
    if hit_ratio == 0:
        correction = 1.0
    elif hit_ratio < 1.0:
        correction = np.log(1. / (1. - hit_ratio)) / hit_ratio
    else:
        correction = 3.0
    return correction

def new_occupancy(pmt_positions, ihit, pmt_separation, row_height=99999, roof_height=0):
    # Vectorised version of occupancy function + check for final row hits
    nearby_hits = 0
    diff = pmt_positions[ihit] - pmt_positions
    nearby_hits = np.count_nonzero(np.sqrt(diff[:,0]**2 +
                                           diff[:,1]**2 +
                                           diff[:,2]**2)
                                   < pmt_separation) - 1
    if np.abs(row_height < pmt_positions[ihit][2] < roof_height):
        # Hit is on top row of barrel, only five neighbours.
        hit_ratio = nearby_hits / 5.
    else:
        hit_ratio = nearby_hits / 8.
    if hit_ratio == 0:
        correction = 1.0
    elif hit_ratio < 1.0:
        correction = np.log(1. / (1. - hit_ratio)) / hit_ratio
    else:
        correction = 3.0
    return correction

def tail_hits(nX, n2X, timeX, dark_noise):
    correction = ((n2X - nX) - (dark_noise * timeX * 1e-9)) / n2X
    return correction

def dark_hits(nX, timeX, dark_noise):
    correction = (dark_noise * timeX * 1e-9) / nX
    return correction

def geometric(points, c_values, ev_vertex):
    vinterp = griddata(points, c_values, ev_vertex, method="linear")
    correction = 1. / vinterp
    return correction

def geometric1D(points, c_values, ev_vertex):
    finterp = interp1d(points, c_values)
    rz_vertex = xyz_to_rz(ev_vertex)
    try:
        correction = finterp(rz_vertex[0])
    except ValueError:
        # Vertex was probably beyond PSUP
        correction = 1.
    return correction

def transparency(pmt_pos, ev_vertex, att_length):
    path_length = fastmag(pmt_pos - ev_vertex)
    correction = np.exp(path_length / att_length)
    return correction

def quantum_efficiency(efficiency):
    correction = 1. / efficiency
    return correction

if __name__ == "__main__":
    estimator = Estimator()
    estimator.parse_options()
    estimator.eventloop()
