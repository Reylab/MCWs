"""
Multi-Method Feature Extraction Pipeline for Spike Data

Provides three feature extraction methods:
1. Haar Wavelet (KS) - Kolmogorov-Smirnov based coefficient selection
2. PCA - Principal Component Analysis
3. GMM - Gaussian Mixture Model (exact MATLAB replication)

Features:
- Process all channels or specific channels
- Automatic method selection
- Progress tracking and detailed logging
- Flexible parameter configuration

Usage:
    from FeatureExtraction import FeatureExtractor
    
    # Using GMM method (recommended)
    extractor = FeatureExtractor(input_dir, output_dir, method='gmm')
    extractor.process_all_channels(channels='all')
    
    # Using Haar/KS method (fast)
    extractor = FeatureExtractor(input_dir, output_dir, method='haar')
    extractor.process_all_channels(channels=[257, 263])
    
    # Using PCA method
    extractor = FeatureExtractor(input_dir, output_dir, method='pca')
    extractor.process_all_channels(channels=290)
"""

import os
import numpy as np
import glob
import traceback
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm
from statsmodels.stats.diagnostic import lilliefors
import pywt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from collections import namedtuple
import warnings

warnings.filterwarnings('ignore')

# Named tuple for GMM results
GMMResult = namedtuple('GMMResult', ['g', 'xg', 'pg', 'med_dist', 'comp_medMax', 
                                     'dist_kj', 'mu_mpdf', 's_mpdf', 'Mcompon', 'polyID', 'Dij'])


class FeatureExtractor:
    """
    Multi-method feature extraction for spike waveforms.
    
    Supports three extraction methods:
    1. Haar/KS (Kolmogorov-Smirnov) - Fast, uses normality testing
    2. PCA - Principal Component Analysis
    3. GMM - Gaussian Mixture Model (exact MATLAB replication)
    
    Parameters:
    -----------
    input_dir : str
        Directory containing *_spikes.mat files
    output_dir : str, optional
        Directory to save extracted features (default: input_dir)
    method : str
        Feature extraction method: 'haar', 'pca', or 'gmm' (default: 'gmm')
    
    Method-specific parameters:
    
    For 'haar' method:
    - scales : int (default: 4) - Wavelet decomposition level
    - max_inputs : float (default: 48) - Max coefficients to consider
    - min_inputs : int (default: 10) - Min coefficients to keep
    - nd : int (default: 10) - Smoothing window for knee detection
    
    For 'pca' method:
    - n_components : int (default: 10) - Number of PCA components
    
    For 'gmm' method:
    - scales : int (default: 4) - Wavelet decomposition level
    - min_weight : float (default: 0.005) - Minimum GMM component weight
    - min_coeff_count : int (default: 5) - Minimum unique coefficients
    - corr_thresh : float (default: 0.9) - Correlation threshold
    
    verbose : bool
        Print detailed progress messages (default: True)
    """
    
    def __init__(self, input_dir, output_dir=None, method='gmm', 
                 scales=4, max_inputs=48, min_inputs=10, nd=10,
                 n_components=10, min_weight=0.005, min_coeff_count=5, 
                 corr_thresh=0.9, verbose=True):
        """Initialize feature extractor."""
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.method = method.lower()
        self.verbose = verbose
        
        # Validate method
        if self.method not in ['haar', 'pca', 'gmm']:
            raise ValueError(f"Invalid method '{method}'. Must be 'haar', 'pca', or 'gmm'")
        
        # Store method-specific parameters
        self.scales = scales
        self.max_inputs = max_inputs
        self.min_inputs = min_inputs
        self.nd = nd
        self.n_components = n_components
        self.min_weight = min_weight
        self.min_coeff_count = min_coeff_count
        self.corr_thresh = corr_thresh
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _log(self, msg):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(msg)
    
    def _parse_channels(self, channels):
        """
        Parse channel specification and return list of channel numbers.
        
        Parameters:
        -----------
        channels : 'all', list, int, or None
            Channel selection specification
            
        Returns:
        --------
        list : Channel numbers to process
        """
        if isinstance(channels, str) and channels.lower() == 'all':
            spike_files = glob.glob(os.path.join(self.input_dir, '*_spikes.mat'))
            channel_nums = []
            for f in spike_files:
                basename = os.path.basename(f)
                parts = basename.replace('_spikes.mat', '').split('_')
                if len(parts) >= 2:
                    try:
                        ch_num = int(parts[-1])
                        channel_nums.append(ch_num)
                    except ValueError:
                        continue
            return sorted(list(set(channel_nums)))
        
        elif isinstance(channels, int):
            return [channels]
        
        elif isinstance(channels, (list, tuple)):
            return sorted(list(set(channels)))
        
        elif channels is None:
            return self._parse_channels('all')
        
        else:
            self._log(f"Warning: Unknown channel type {type(channels)}, using all channels")
            return self._parse_channels('all')
    
    def _find_spike_file(self, channel_num):
        """Find spike file for a specific channel."""
        spike_files = glob.glob(os.path.join(self.input_dir, '*_spikes.mat'))
        for f in spike_files:
            basename = os.path.basename(f)
            if basename.endswith(f'{channel_num}_spikes.mat'):
                return f
        return None
    
    # ========================================================================
    # HAAR/KS METHOD
    # ========================================================================
    
    def _extract_haar(self, spikes):
        """
        Extract features using Haar wavelet + Kolmogorov-Smirnov method.
        Based on feature_extraction.py haar_feature_extraction_for_channel.
        
        Parameters:
        -----------
        spikes : ndarray
            Spike waveforms (n_spikes, spike_length)
            
        Returns:
        --------
        features : ndarray
            Selected wavelet coefficients (n_spikes, n_selected)
        selected_coeffs : ndarray
            Selected coefficient indices
        summary_info : dict
            Extraction summary information
        """
        nspk, ls = spikes.shape
        
        self._log(f"  Running Haar wavelet decomposition (level={self.scales})...")
        
        # Compute Haar coefficients for each spike
        cc = np.zeros((nspk, ls))
        
        for i in range(nspk):
            c = pywt.wavedec(spikes[i], 'haar', level=self.scales)
            flattened_coeffs = np.hstack(c)[:ls]
            cc[i, :len(flattened_coeffs)] = flattened_coeffs
        
        ls = cc.shape[1]
        
        self._log(f"  Computing Kolmogorov-Smirnov normality test...")
        
        # KS normality test on each coefficient
        ks = np.zeros(ls)
        
        for i in range(ls):
            thr_dist = np.std(cc[:, i]) * 3
            thr_dist_min = np.mean(cc[:, i]) - thr_dist
            thr_dist_max = np.mean(cc[:, i]) + thr_dist
            aux = cc[(cc[:, i] > thr_dist_min) & (cc[:, i] < thr_dist_max), i]
            
            if len(aux) > 10:
                try:
                    ks[i] = lilliefors(aux, dist='norm', pvalmethod='table')[0]
                except:
                    ks[i] = 0
            else:
                ks[i] = 0
        
        # Sort by KS statistic (higher = more normal = less interesting)
        sorted_indices = np.argsort(ks)
        A = ks[sorted_indices]
        
        self._log(f"  Applying knee detection...")
        
        # Knee detection on top candidates
        ncoeff = len(A)
        maxA = np.max(A) if len(A) > 0 else 1
        
        nd = self.nd
        max_inputs = int(self.max_inputs)
        min_inputs = int(self.min_inputs)
        
        if ncoeff >= nd:
            d = ((A[nd-1:] - A[:-nd+1]) / maxA) * (ncoeff / nd)
            all_above1 = np.where(d >= 1)[0]
        else:
            all_above1 = np.array([])
        
        if len(all_above1) >= 2:
            aux2 = np.diff(all_above1)
            temp_bla_full = np.convolve(aux2, np.array([1, 1, 1]) / 3, mode='full')
            
            if len(aux2) - 1 == 1:
                temp_bla = np.array([temp_bla_full[1]])
                temp_bla[0] = aux2[0]
            elif len(aux2) - 1 == 0:
                temp_bla = np.array([aux2[0]])
            else:
                temp_bla = temp_bla_full[1:len(aux2)+1]
                temp_bla[0] = aux2[0]
                temp_bla[-1] = aux2[-1]
            
            if not temp_bla[1:].size:
                thr_knee_diff = 0
            else:
                knee_idx = np.where(temp_bla[1:] == 1)[0]
                if len(knee_idx) > 0:
                    thr_knee_diff = all_above1[knee_idx[0]] + (nd / 2)
                else:
                    thr_knee_diff = 0
            
            inputs = max_inputs - thr_knee_diff
        else:
            inputs = min_inputs
        
        # Clamp inputs
        if inputs > max_inputs:
            inputs = max_inputs
        elif inputs < min_inputs:
            inputs = min_inputs
        
        inputs = int(inputs)
        
        # Select coefficients
        coeff = sorted_indices[ls-1:ls-inputs-1:-1]
        features = np.zeros((nspk, inputs))
        
        for i in range(nspk):
            for j in range(inputs):
                features[i, j] = cc[i, coeff[j]]
        
        summary_info = {
            'method': 'haar',
            'selected_coeffs': coeff,
            'num_coeffs': inputs,
            'ks_stats': ks,
            'all_coeffs': cc
        }
        
        self._log(f"  Selected {inputs} coefficients")
        
        return features, coeff, summary_info
    
    # ========================================================================
    # PCA METHOD
    # ========================================================================
    
    def _extract_pca(self, spikes):
        """
        Extract features using Principal Component Analysis.
        
        Parameters:
        -----------
        spikes : ndarray
            Spike waveforms (n_spikes, spike_length)
            
        Returns:
        --------
        features : ndarray
            PCA transformed features (n_spikes, n_components)
        selected_coeffs : ndarray
            Component indices (0 to n_components-1)
        summary_info : dict
            Extraction summary information
        """
        nspk = spikes.shape[0]
        n_comp = min(self.n_components, spikes.shape[0], spikes.shape[1])
        
        self._log(f"  Fitting PCA with {n_comp} components...")
        
        pca = PCA(n_components=n_comp)
        features = pca.fit_transform(spikes)
        
        self._log(f"  Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        selected_coeffs = np.arange(0, n_comp)
        
        summary_info = {
            'method': 'pca',
            'selected_coeffs': selected_coeffs,
            'num_coeffs': n_comp,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_
        }
        
        return features, selected_coeffs, summary_info
    
    # ========================================================================
    # GMM METHOD
    # ========================================================================
    
    def _haar_coeffs_gmm(self, spikes):
        """Compute Haar wavelet coefficients for GMM."""
        nspk, ls = spikes.shape
        cc = np.zeros((nspk, ls))
        
        try:
            spikes_l = spikes.flatten()
            coeffs = pywt.wavedec(spikes_l, 'haar', level=self.scales)
            c_l = np.concatenate(coeffs)
            l_wc = np.array([len(c) for c in coeffs])
            wv_c = np.concatenate(([0], l_wc[:-1]))
            nc = wv_c / nspk
            wccum = np.cumsum(wv_c)
            nccum = np.cumsum(nc)
            
            for cf in range(1, len(nc)):
                start_idx = int(nccum[cf-1])
                end_idx = int(nccum[cf])
                start_c = int(wccum[cf-1])
                end_c = int(wccum[cf])
                n_coeff = int(nc[cf])
                cc[:, start_idx:end_idx] = c_l[start_c:end_c].reshape((n_coeff, nspk), order='F').T
        except:
            for i in range(nspk):
                coeffs = pywt.wavedec(spikes[i, :], 'haar', level=self.scales)
                c = np.concatenate(coeffs)
                cc[i, :ls] = c[:ls]
        
        return cc
    
    def _gmm_basic1d(self, x):
        """Fit GMM to 1D feature vector and compute metrics."""
        x = x.flatten()
        
        K = 16
        Mgrid = 100
        n_replicates = 5
        
        best_gmm = None
        best_bic = np.inf
        
        for rep in range(n_replicates):
            try:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type='diag',
                    max_iter=3000,
                    tol=1e-5,
                    reg_covar=1e-5,
                    init_params='k-means++',
                    random_state=None
                )
                gmm.fit(x.reshape(-1, 1))
                bic = gmm.bic(x.reshape(-1, 1))
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except:
                continue
        
        if best_gmm is None:
            return None
        
        mu = best_gmm.means_.flatten()
        s = np.sqrt(best_gmm.covariances_.flatten())
        a = best_gmm.weights_.flatten()
        polyID = np.arange(1, len(mu) + 1)
        
        xg = np.linspace(x.min(), x.max(), Mgrid)
        pg = np.exp(best_gmm.score_samples(xg.reshape(-1, 1)))
        
        peak_indices, _ = find_peaks(pg, height=0)
        
        if len(peak_indices) > 0:
            widths_result = peak_widths(pg, peak_indices, rel_height=0.5)
            pk_vals = pg[peak_indices]
            tallest_idx_in_peaks = np.argmax(pk_vals)
            main_peak_idx = peak_indices[tallest_idx_in_peaks]
            main_peak_center = xg[main_peak_idx]
            
            left_bound_idx = widths_result[2][tallest_idx_in_peaks]
            right_bound_idx = widths_result[3][tallest_idx_in_peaks]
            
            left_bound = np.interp(left_bound_idx, np.arange(len(xg)), xg)
            right_bound = np.interp(right_bound_idx, np.arange(len(xg)), xg)
            main_peak_width = right_bound - left_bound
        else:
            main_peak_center = np.mean(mu)
            main_peak_width = 2 * np.std(x)
        
        left_bound = main_peak_center - main_peak_width / 2
        right_bound = main_peak_center + main_peak_width / 2
        
        in_main_peak = (mu >= left_bound) & (mu <= right_bound)
        a_top_idx = np.where(in_main_peak)[0]
        mu_sort = mu[a_top_idx]
        s_sort = s[a_top_idx]
        polyID_M = polyID[a_top_idx]
        a_top = a[a_top_idx]
        
        M_pdf = np.zeros_like(xg)
        
        if len(a_top) > 0:
            scale_a = a_top / np.sum(a_top)
            for i in range(len(a_top)):
                Npdf_i = norm.pdf(xg, mu_sort[i], s_sort[i])
                M_pdf += scale_a[i] * Npdf_i
        
        Knew = len(mu)
        
        if np.sum(M_pdf) > 0:
            mu_M = np.sum(xg * M_pdf) / np.sum(M_pdf)
            std_M = np.sqrt(np.sum((xg - mu_M)**2 * M_pdf) / np.sum(M_pdf))
        else:
            mu_M = np.mean(mu)
            std_M = np.std(mu)
        
        if std_M > 0:
            kj = np.abs(mu - mu_M) / std_M
        else:
            kj = np.zeros_like(mu)
        
        Dij_list = []
        II_list = []
        JJ_list = []
        
        for i in range(Knew):
            for j in range(i + 1, Knew):
                dij = (np.abs(mu[i] - mu[j]) * np.sqrt(a[i] * a[j]) / 
                       np.sqrt(s[i] * s[j]))
                Dij_list.append(dij)
                II_list.append(i)
                JJ_list.append(j)
        
        Dij = np.array(Dij_list) if Dij_list else np.array([])
        
        med_dist = np.zeros(K)
        for comp_idx in range(Knew):
            comp_distances = []
            for idx, (i, j) in enumerate(zip(II_list, JJ_list)):
                if i == comp_idx or j == comp_idx:
                    comp_distances.append(Dij[idx])
            if comp_distances:
                med_dist[comp_idx] = np.median(comp_distances)
        
        comp_medMax = np.argmax(med_dist[:Knew]) + 1 if Knew > 0 else 1
        
        return GMMResult(
            g=best_gmm,
            xg=xg,
            pg=pg,
            med_dist=med_dist[:Knew],
            comp_medMax=comp_medMax,
            dist_kj=kj,
            mu_mpdf=mu_M,
            s_mpdf=std_M,
            Mcompon=polyID_M,
            polyID=polyID,
            Dij=Dij
        )
    
    def _gmm_1channel(self, spikes):
        """Main GMM feature extraction function."""
        coeffs = self._haar_coeffs_gmm(spikes)
        m = coeffs.shape[1]
        
        medDist = [[None, None] for _ in range(m)]
        g = [None] * m
        xg = [None] * m
        pg = [None] * m
        summary_gmm = [[None] * m for _ in range(10)]
        
        for j in range(m):
            Mj = self._gmm_basic1d(coeffs[:, j])
            
            if Mj is None:
                continue
            
            summary_gmm[0][j] = j + 1
            summary_gmm[1][j] = Mj.dist_kj
            summary_gmm[2][j] = Mj.Dij
            summary_gmm[3][j] = Mj.med_dist
            summary_gmm[4][j] = Mj.mu_mpdf
            summary_gmm[5][j] = Mj.s_mpdf
            summary_gmm[6][j] = Mj.g.means_.flatten()
            summary_gmm[7][j] = Mj.g.covariances_.flatten()
            summary_gmm[8][j] = Mj.Mcompon
            summary_gmm[9][j] = Mj.polyID
            
            medDist[j][0] = Mj.med_dist
            medDist[j][1] = Mj.comp_medMax
            
            g[j] = Mj.g
            xg[j] = Mj.xg
            pg[j] = Mj.pg
        
        summary_out = {
            'coeff_num': summary_gmm[0],
            'kj': summary_gmm[1],
            'Dij': summary_gmm[2],
            'med_idist': summary_gmm[3],
            'mpdf_mu': summary_gmm[4],
            'mpdf_std': summary_gmm[5],
            'mu_gmm': summary_gmm[6],
            'std_gmm': summary_gmm[7],
            'M_comp': summary_gmm[8],
            'polyID': summary_gmm[9]
        }
        
        return g, xg, pg, coeffs, medDist, summary_out
    
    def _process_knee(self, sorted_vals, sorted_idx, sorted_gauss):
        """Apply knee detection to select top values."""
        maxVal = len(sorted_vals)
        
        ncoeff = len(sorted_vals)
        maxA = np.max(sorted_vals) if len(sorted_vals) > 0 else 1
        
        nd = max(10, int(round(maxVal / 10)))
        
        if ncoeff >= nd:
            d_vals = ((sorted_vals[nd-1:] - sorted_vals[:ncoeff-nd+1]) / 
                      maxA * ncoeff / nd)
            all_above1 = np.where(d_vals >= 1)[0]
        else:
            all_above1 = np.array([])
        
        if len(all_above1) >= 2:
            aux2 = np.diff(all_above1)
            temp_bla_full = np.convolve(aux2, np.array([1, 1, 1]) / 3, mode='full')
            temp_bla = temp_bla_full[1:-1]
            
            if len(temp_bla) > 0:
                temp_bla[0] = aux2[0]
                temp_bla[-1] = aux2[-1]
            
            idx_ones = np.where(temp_bla[1:] == 1)[0]
            
            if len(idx_ones) > 0:
                thr_knee_diff = all_above1[idx_ones[0]] + int(round(nd / 2))
                inputs = maxVal - thr_knee_diff + 1
            else:
                inputs = min(10, maxVal)
        else:
            inputs = min(10, maxVal)
        
        inputs = max(1, min(int(inputs), maxVal))
        
        select_array = np.zeros((inputs, 3))
        select_array[:, 0] = sorted_vals[-inputs:][::-1]
        select_array[:, 1] = sorted_idx[-inputs:][::-1]
        select_array[:, 2] = sorted_gauss[-inputs:][::-1]
        
        return select_array
    
    def _extract_gmm(self, spikes):
        """
        Extract features using Gaussian Mixture Model.
        Exact MATLAB replication from GMM.M
        
        Parameters:
        -----------
        spikes : ndarray
            Spike waveforms (n_spikes, spike_length)
            
        Returns:
        --------
        features : ndarray
            Selected wavelet coefficients (n_spikes, n_selected)
        selected_coeffs : ndarray
            Selected coefficient indices
        summary_info : dict
            Extraction summary information
        """
        self._log(f"  Running GMM_1channel (scales={self.scales})...")
        g, xg, pg, wd_coeffs, medDist, summary_table = self._gmm_1channel(spikes)
        
        m = len(g)
        self._log(f"  Processed {m} coefficients")
        
        self._log(f"  Filtering components (min_weight={self.min_weight})...")
        
        medDist_cell = []
        kj_mat = []
        polyID_filtered = []
        M_comp = []
        idistVals = np.zeros(m)
        max_k = np.zeros(m)
        
        medDist_init = summary_table['med_idist']
        kj_init = summary_table['kj']
        Mcomp_init = summary_table['M_comp']
        
        for i in range(m):
            if g[i] is None:
                medDist_cell.append(np.array([]))
                kj_mat.append(np.array([]))
                polyID_filtered.append(np.array([]))
                M_comp.append(np.array([]))
                continue
            
            g_in = g[i]
            mu = g_in.means_.flatten()
            s = np.sqrt(g_in.covariances_.flatten())
            a = g_in.weights_.flatten()
            polyID_in = np.arange(1, len(mu) + 1)
            
            keep = a > self.min_weight
            
            polyID_keep = polyID_in[keep]
            
            medDist_filt = medDist_init[i][keep]
            kj_filt = kj_init[i][keep]
            
            M_comp_i = Mcomp_init[i]
            M_comp_keep = M_comp_i[np.isin(M_comp_i, polyID_keep)]
            M_comp.append(M_comp_keep)
            
            exclude_mask = np.isin(polyID_keep, M_comp_keep)
            
            medDist_masked = medDist_filt[~exclude_mask]
            kj_masked = kj_filt[~exclude_mask]
            polyID_masked = polyID_keep[~exclude_mask]
            
            medDist_cell.append(medDist_masked)
            kj_mat.append(kj_masked)
            polyID_filtered.append(polyID_masked)
            
            if len(medDist_masked) > 0:
                idistVals[i] = np.max(medDist_masked)
                max_k[i] = np.max(kj_masked)
        
        self._log(f"  Expanding metrics into vectors...")
        
        k_vec_list = []
        for k in range(m):
            kV_k = kj_mat[k]
            kv_Gidx = polyID_filtered[k]
            for val, gauss_id in zip(kV_k, kv_Gidx):
                if val != 0:
                    k_vec_list.append([val, k, gauss_id])
        
        k_vector = np.array(k_vec_list) if k_vec_list else np.zeros((0, 3))
        
        medDist_vec_list = []
        for k in range(m):
            medDist_k = medDist_cell[k]
            medDist_Gidx = polyID_filtered[k]
            for val, gauss_id in zip(medDist_k, medDist_Gidx):
                if val != 0:
                    medDist_vec_list.append([val, k, gauss_id])
        
        medDist_vector = np.array(medDist_vec_list) if medDist_vec_list else np.zeros((0, 3))
        
        if len(k_vector) > 0:
            sort_idx_k = np.argsort(k_vector[:, 0])
            kDist_vec = k_vector[sort_idx_k]
        else:
            kDist_vec = np.zeros((0, 3))
        
        if len(medDist_vector) > 0:
            sort_idx_med = np.argsort(medDist_vector[:, 0])
            medDist_vec = medDist_vector[sort_idx_med]
        else:
            medDist_vec = np.zeros((0, 3))
        
        self._log(f"  Applying knee detection...")
        
        if len(kDist_vec) > 0:
            k_select = self._process_knee(
                kDist_vec[:, 0],
                kDist_vec[:, 1],
                kDist_vec[:, 2]
            )
        else:
            k_select = np.zeros((0, 3))
        
        if len(medDist_vec) > 0:
            medDist_select = self._process_knee(
                medDist_vec[:, 0],
                medDist_vec[:, 1],
                medDist_vec[:, 2]
            )
        else:
            medDist_select = np.zeros((0, 3))
        
        self._log(f"  Selecting final coefficients...")
        
        if len(medDist_select) > 0 and len(k_select) > 0:
            medDist_coeffs = medDist_select[:, 1].astype(int)
            k_coeffs = k_select[:, 1].astype(int)
            
            all_coeffs = np.unique(np.concatenate([medDist_coeffs, k_coeffs]))
            
            n_select = min(10, len(all_coeffs))
            
            coeff_scores = []
            for coeff_idx in all_coeffs:
                medDist_score = idistVals[coeff_idx]
                kj_score = max_k[coeff_idx]
                combined_score = medDist_score + kj_score
                coeff_scores.append((coeff_idx, combined_score))
            
            coeff_scores.sort(key=lambda x: x[1], reverse=True)
            selected_coeffs = np.array([c[0] for c in coeff_scores[:n_select]]).astype(int)
        else:
            combined_metric = idistVals + max_k
            selected_coeffs = np.argsort(combined_metric)[-10:]
        
        features = wd_coeffs[:, selected_coeffs]
        
        summary_info = {
            'method': 'gmm',
            'selected_coeffs': selected_coeffs,
            'num_coeffs': len(selected_coeffs),
            'g_init': g,
            'medDist_vec': medDist_vec,
            'kDist_vec': kDist_vec,
            'idistVals': idistVals,
            'max_k': max_k,
            'all_coeffs': wd_coeffs
        }
        
        self._log(f"  Selected {len(selected_coeffs)} coefficients")
        
        return features, selected_coeffs, summary_info
    
    # ========================================================================
    # MAIN EXTRACTION DISPATCHER
    # ========================================================================
    
    def _extract_features(self, spikes):
        """
        Dispatch to appropriate feature extraction method.
        
        Parameters:
        -----------
        spikes : ndarray
            Spike waveforms
            
        Returns:
        --------
        features : ndarray
            Extracted features
        selected_coeffs : ndarray
            Selected coefficient/component indices
        summary_info : dict
            Extraction summary
        """
        if self.method == 'haar':
            return self._extract_haar(spikes)
        elif self.method == 'pca':
            return self._extract_pca(spikes)
        elif self.method == 'gmm':
            return self._extract_gmm(spikes)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def process_single_channel(self, channel_num):
        """
        Process a single channel's spike data.
        
        Parameters:
        -----------
        channel_num : int
            Channel number to process
            
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        spike_file = self._find_spike_file(channel_num)
        
        if spike_file is None:
            self._log(f"Warning: No spike file found for channel {channel_num}")
            return False
        
        try:
            self._log(f"\n{'='*60}")
            self._log(f"Channel {channel_num} - {self.method.upper()} extraction")
            self._log(f"{'='*60}")
            
            # Load spike data
            mat_data = loadmat(spike_file)
            
            # Extract spikes
            if 'spikes' in mat_data:
                spikes = mat_data['spikes'].astype(np.float64)
            elif 'spikes_all' in mat_data:
                spikes = mat_data['spikes_all'].astype(np.float64)
            else:
                self._log("Error: No 'spikes' or 'spikes_all' field found")
                return False
            
            num_spikes = spikes.shape[0]
            self._log(f"Loaded {num_spikes} spikes (shape: {spikes.shape})")
            
            if num_spikes == 0:
                self._log("No spikes to process. Skipping.")
                return False
            
            # Extract features
            features, selected_coeffs, summary_info = self._extract_features(spikes)
            
            # Prepare output
            output_data = {
                'spikes': spikes,
                'spikes_features': features,
                'selected_coeffs': selected_coeffs,
                'num_spikes': num_spikes,
                'num_features': features.shape[1],
                'method': self.method
            }
            
            # Add original fields if they exist
            for key in mat_data.keys():
                if key.startswith('__') or key.startswith('_'):
                    continue
                if key not in output_data:
                    output_data[key] = mat_data[key]
            
            # Save output
            output_filename = f"ch{channel_num}_features_{self.method}.mat"
            output_path = os.path.join(self.output_dir, output_filename)
            savemat(output_path, output_data, do_compression=True)
            
            self._log(f"Saved features to: {output_filename}")
            self._log(f"Feature matrix shape: {features.shape}")
            
            return True
            
        except Exception as e:
            self._log(f"ERROR processing channel {channel_num}: {e}")
            traceback.print_exc()
            return False
    
    def process_all_channels(self, channels='all'):
        """
        Process spike data for specified channels.
        
        Parameters:
        -----------
        channels : 'all', list, int, or None
            Channels to process:
            - 'all': Process all channels (default)
            - [257, 263]: Process specific channels
            - 290: Process single channel
            - None: Process all channels
        
        Examples:
        ---------
        # Process all channels with GMM
        extractor = FeatureExtractor(input_dir, output_dir, method='gmm')
        extractor.process_all_channels(channels='all')
        
        # Process specific channels with Haar
        extractor = FeatureExtractor(input_dir, output_dir, method='haar')
        extractor.process_all_channels(channels=[257, 263, 290])
        
        # Process range with PCA
        extractor = FeatureExtractor(input_dir, output_dir, method='pca')
        extractor.process_all_channels(channels=list(range(290, 301)))
        """
        # Parse channels
        channel_nums = self._parse_channels(channels)
        
        if not channel_nums:
            self._log(f"No channels specified or found in {self.input_dir}")
            return
        
        # Summary header
        self._log(f"\n{'#'*70}")
        self._log(f"FEATURE EXTRACTION PIPELINE")
        self._log(f"{'#'*70}")
        self._log(f"Method: {self.method.upper()}")
        self._log(f"Input directory: {self.input_dir}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Channels: {channel_nums}")
        self._log(f"Total channels: {len(channel_nums)}")
        
        # Method-specific parameters
        if self.method == 'haar':
            self._log(f"Scales: {self.scales}, Max inputs: {self.max_inputs}, Min inputs: {self.min_inputs}")
        elif self.method == 'pca':
            self._log(f"Components: {self.n_components}")
        elif self.method == 'gmm':
            self._log(f"Scales: {self.scales}, Min weight: {self.min_weight}, Min coeff: {self.min_coeff_count}")
        
        self._log(f"{'#'*70}\n")
        
        # Process each channel
        successful = 0
        failed = 0
        
        for i, ch_num in enumerate(channel_nums, 1):
            self._log(f"[{i}/{len(channel_nums)}] Processing channel {ch_num}...")
            
            if self.process_single_channel(ch_num):
                successful += 1
            else:
                failed += 1
        
        # Summary footer
        self._log(f"\n{'#'*70}")
        self._log(f"PROCESSING COMPLETE")
        self._log(f"{'#'*70}")
        self._log(f"Successful: {successful}/{len(channel_nums)}")
        self._log(f"Failed: {failed}/{len(channel_nums)}")
        self._log(f"Output files: {self.output_dir}")
        self._log(f"{'#'*70}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of FeatureExtractor with different methods
    """
    
    # Define directories
    input_dir = '/media/sEEG_DATA/Tests/Matlab sorting pipeline/Tapasi/MCWs_Pipeline/full_processing_pipeline/output'
    output_dir = '/media/sEEG_DATA/Tests/Matlab sorting pipeline/Tapasi/MCWs_Pipeline/full_processing_pipeline/output/features'
    
    # ========================================
    # EXAMPLE 1: GMM EXTRACTION (RECOMMENDED)
    # ========================================
    print("\n" + "="*70)
    print("EXAMPLE 1: GMM METHOD (Exact MATLAB replication)")
    print("="*70)
    
    extractor_gmm = FeatureExtractor(
        input_dir=input_dir,
        output_dir=output_dir,
        method='gmm',
        scales=4,
        min_weight=0.005,
        min_coeff_count=5,
        corr_thresh=0.9,
        verbose=True
    )
    
    # Process all channels
    # extractor_gmm.process_all_channels(channels='all')
    
    # Process specific channels
    # extractor_gmm.process_all_channels(channels=[257, 263])
    
    # Process range
    extractor_gmm.process_all_channels(channels=list(range(290, 301)))
    
    # ========================================
    # EXAMPLE 2: HAAR/KS EXTRACTION (FAST)
    # ========================================
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: HAAR/KS METHOD (Fast)")
    print("="*70)
    
    extractor_haar = FeatureExtractor(
        input_dir=input_dir,
        output_dir=output_dir,
        method='haar',
        scales=4,
        max_inputs=48,
        min_inputs=10,
        nd=10,
        verbose=True
    )
    
    extractor_haar.process_all_channels(channels=[257, 263])
    """
    
    # ========================================
    # EXAMPLE 3: PCA EXTRACTION
    # ========================================
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: PCA METHOD")
    print("="*70)
    
    extractor_pca = FeatureExtractor(
        input_dir=input_dir,
        output_dir=output_dir,
        method='pca',
        n_components=10,
        verbose=True
    )
    
    extractor_pca.process_all_channels(channels='all')
    """