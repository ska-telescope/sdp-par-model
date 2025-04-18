"""
Enumerates and defines the parameters of the telescopes, bands,
pipelines, etc. Several methods are supplied by which values can be
found by lookup as well (e.g. finding the telescope that is associated
with a given mode)

Parameters defined here include telecope parameters as well as
physical constants. Generally, the output of the functions are
ParameterContainer objects (usually referred to by the variable o in
the methods below) that has parameters as fields.
"""

from __future__ import print_function

import itertools
import warnings

import numpy as np
from sympy import symbols

from .container import ParameterContainer


class Constants:
    """
    A new class that takes over the roles of sympy.physics.units and
    astropy.const, because it is simpler this way.
    """

    kilo = 1000
    mega = 1000000
    giga = 1000000000
    tera = 1000000000000
    peta = 1000000000000000

    degree = np.pi / 180
    arcminute = np.pi / 180 / 60
    arcsecond = np.pi / 180 / 60 / 60


class Telescopes:
    """
    Enumerate the possible telescopes to choose from (used in
    :meth:`apply_telescope_parameters`)
    """

    SKA1_Low = "SKA1_Low"
    SKA1_Low_AA05 = "SKA1_Low_AA05"
    SKA1_Low_AA1 = "SKA1_Low_AA1"
    SKA1_Low_AA2 = "SKA1_Low_AA2"
    SKA1_Low_AA15p = "SKA1_Low_AA15p"
    SKA1_Low_AA2p = "SKA1_Low_AA2p"
    SKA1_Low_AA3 = "SKA1_Low_AA3"
    SKA1_Mid = "SKA1_Mid"
    SKA1_Mid_AA1 = "SKA1_Mid_AA1"
    SKA1_Mid_AA2 = "SKA1_Mid_AA2"
    SKA1_Mid_AAs = "SKA1_Mid_AAs"
    LOFAR_LBA = "LOFAR_LBA"
    LOFAR_HBA = "LOFAR_HBA"

    # Currently supported telescopes (will show up in notebooks)
    available_teles = [
        SKA1_Low,
        SKA1_Low_AA05,
        SKA1_Low_AA1,
        SKA1_Low_AA2,
        SKA1_Low_AA15p,
        SKA1_Low_AA2p,
        SKA1_Low_AA3,
        SKA1_Mid,
        SKA1_Mid_AA2,
        SKA1_Mid_AAs,
        LOFAR_HBA,
    ]


class Bands:
    """
    Enumerate all possible bands (used in :meth:`apply_band_parameters`)
    """

    # SKA1 Bands
    Low = "Low"
    LowAA1 = "LowAA1"
    LowAA2 = "LowAA2"
    Mid1 = "Mid1"
    Mid2 = "Mid2"
    Mid5a = "Mid5a"
    Mid5b = "Mid5b"
    LofarHigh1 = "LofarHigh1"
    LofarHigh2 = "LofarHigh2"
    LofarHigh3 = "LofarHigh3"

    # group the bands defined above into logically coherent sets
    telescope_bands = {
        Telescopes.SKA1_Low: [Low],
        Telescopes.SKA1_Low_AA05: [LowAA1],
        Telescopes.SKA1_Low_AA1: [LowAA1],
        Telescopes.SKA1_Low_AA2: [LowAA2],
        Telescopes.SKA1_Low_AA15p: [LowAA1],
        Telescopes.SKA1_Low_AA2p: [LowAA2],
        Telescopes.SKA1_Low_AA3: [Low],
        Telescopes.SKA1_Mid: [Mid1, Mid2, Mid5a, Mid5b],
        Telescopes.SKA1_Mid_AA2: [Mid1, Mid2, Mid5a, Mid5b],
        Telescopes.SKA1_Mid_AAs: [Mid1, Mid2, Mid5a, Mid5b],
        Telescopes.LOFAR_HBA: [LofarHigh1, LofarHigh2, LofarHigh3],
    }
    available_bands = list(
        itertools.chain.from_iterable(telescope_bands.values())
    )


class Products:
    """
    Enumerate the SDP Products used in pipelines
    """

    Alert = "Alert"
    Average = "Average"
    Calibration_Source_Finding = "Calibration Source Finding"
    Correct = "Correct"
    Degrid = "Degrid"
    DFT = "DFT"
    Demix = "Demix"
    FFT = "FFT"
    Flag = "Flag"
    Grid = "Grid"
    Gridding_Kernel_Update = "Gridding Kernel Update"
    Degridding_Kernel_Update = "Degridding Kernel Update"
    Identify_Component = "Identify Component"
    Extract_LSM = "Extract_LSM"
    IFFT = "IFFT"
    Image_Spectral_Averaging = "Image Spectral Averaging"
    Image_Spectral_Fitting = "Image Spectral Fitting"
    Notify_GSM = "Update GSM"
    PhaseRotation = "Phase Rotation"
    PhaseRotationPredict = "Phase Rotation Predict"
    QA = "QA"
    Receive = "Receive"
    Reprojection = "Reprojection"
    ReprojectionPredict = "Reprojection Predict"
    Select = "Select"
    Solve = "Solve"
    Source_Find = "Source Find"
    Subtract_Visibility = "Subtract Visibility"
    Subtract_Image_Component = "Subtract Image Component"
    Update_LSM = "Update LSM"
    Visibility_Weighting = "Visibility Weighting"


class Pipelines:
    """
    Enumerate the SDP pipelines. These must map onto the Products. The HPSOs invoke these.
    """

    Ingest = "Ingest"  # Ingest data
    RCAL = "RCAL"  # Produce calibration solutions in real time
    FastImg = (
        "FastImg"  # Produce continuum subtracted residual image every 1s or so
    )
    ICAL = "ICAL"  # Produce calibration solutions using iterative self-calibration
    DPrepA = "DPrepA"  # Produce continuum Taylor term images in Stokes I
    DPrepA_Image = "DPrepA_Image"  # Produce continuum Taylor term images in Stokes I as CASA does in images
    DPrepB = "DPrepB"  # Produce coarse continuum image cubes in I,Q,U,V (with Nf_out channels)
    DPrepC = "DPrepC"  # Produce fine spectral resolution image cubes un I,Q,U,V (with Nf_out channels)
    DPrepD = "DPrepD"  # Produce calibrated, averaged (In time and freq) visibility data

    PSS = "PSS"  # Sieved Pulsars
    PST = "PST"  # Pulsar Timing Solutions
    SinglePulse = "SinglePulse"  # Transient Buffer Data

    input = [Ingest]
    realtime = [Ingest, RCAL, FastImg]
    imaging = [RCAL, FastImg, ICAL, DPrepA, DPrepA_Image, DPrepB, DPrepC]
    nonimaging = [PSS, PST, SinglePulse]
    output = [
        FastImg,
        DPrepA,
        DPrepA_Image,
        DPrepB,
        DPrepC,
        PSS,
        PST,
        SinglePulse,
    ]
    all = [
        Ingest,
        RCAL,
        FastImg,
        ICAL,
        DPrepA,
        DPrepA_Image,
        DPrepB,
        DPrepC,
        DPrepD,
        PSS,
        PST,
        SinglePulse,
    ]
    pure_pipelines = all

    # Pipelines that are currently supported (will show up in notebooks)
    available_pipelines = all


class HPSOs:
    """
    Enumerate the pipelines of each HPSO (used in :meth:`apply_hpso_parameters`)
    """

    # The high-priority science objectives (HPSOs).

    hpso01 = "hpso01"
    hpso02a = "hpso02a"
    hpso02b = "hpso02b"
    hpso04a = "hpso04a"
    hpso04b = "hpso04b"
    hpso04c = "hpso04c"
    hpso05a = "hpso05a"
    hpso05b = "hpso05b"
    hpso13 = "hpso13"
    hpso14 = "hpso14"
    hpso15 = "hpso15"
    hpso18 = "hpso18"
    hpso22 = "hpso22"
    hpso27and33 = "hpso27and33"
    hpso32 = "hpso32"
    hpso37a = "hpso37a"
    hpso37b = "hpso37b"
    hpso37c = "hpso37c"
    hpso38a = "hpso38a"
    hpso38b = "hpso38b"

    # Maximal cases for the telescopes. For Mid, define a maximal case
    # for each of the bands. Mid bands 5a and 5b allow a bandwidth of
    # up to 2.5 GHz to be observed simultaneously, so define two cases
    # for each of them corresponding to the lowest and highest 2.5
    # GHz of the band.

    max_low = "max_low"
    max_mid_band1 = "max_mid_band1"
    max_mid_band2 = "max_mid_band2"
    max_mid_band5a_1 = "max_mid_band5a_1"
    max_mid_band5a_2 = "max_mid_band5a_2"
    max_mid_band5b_1 = "max_mid_band5b_1"
    max_mid_band5b_2 = "max_mid_band5b_2"

    hpso_telescopes = {
        hpso01: Telescopes.SKA1_Low_AA3,
        hpso02a: Telescopes.SKA1_Low_AA3,
        hpso02b: Telescopes.SKA1_Low_AA3,
        hpso04a: Telescopes.SKA1_Low_AA3,
        hpso04b: Telescopes.SKA1_Mid,
        hpso04c: Telescopes.SKA1_Mid,
        hpso05a: Telescopes.SKA1_Low_AA3,
        hpso05b: Telescopes.SKA1_Mid,
        hpso13: Telescopes.SKA1_Mid,
        hpso14: Telescopes.SKA1_Mid,
        hpso15: Telescopes.SKA1_Mid,
        hpso18: Telescopes.SKA1_Mid,
        hpso22: Telescopes.SKA1_Mid,
        hpso27and33: Telescopes.SKA1_Mid,
        hpso32: Telescopes.SKA1_Mid,
        hpso37a: Telescopes.SKA1_Mid,
        hpso37b: Telescopes.SKA1_Mid,
        hpso37c: Telescopes.SKA1_Mid,
        hpso38a: Telescopes.SKA1_Mid,
        hpso38b: Telescopes.SKA1_Mid,
        max_low: Telescopes.SKA1_Low,
        max_mid_band1: Telescopes.SKA1_Mid,
        max_mid_band2: Telescopes.SKA1_Mid,
        max_mid_band5a_1: Telescopes.SKA1_Mid,
        max_mid_band5a_2: Telescopes.SKA1_Mid,
        max_mid_band5b_1: Telescopes.SKA1_Mid,
        max_mid_band5b_2: Telescopes.SKA1_Mid,
    }

    # Map each HPSO to its constituent pipelines

    hpso_pipelines = {
        hpso01: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        hpso02a: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        hpso02b: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        hpso04a: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.PSS,
        ),
        hpso04b: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.PSS,
        ),
        hpso04c: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.PSS,
        ),
        hpso05a: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.PST,
        ),
        hpso05b: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.PST,
        ),
        hpso13: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
        ),
        hpso14: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
        ),
        hpso15: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
        ),
        hpso18: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.SinglePulse,
        ),
        hpso22: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        hpso27and33: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        hpso32: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepB,
        ),
        hpso37a: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        hpso37b: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        hpso37c: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        hpso38a: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        hpso38b: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepB,
        ),
        max_low: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        max_mid_band1: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        max_mid_band2: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        max_mid_band5a_1: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        max_mid_band5a_2: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        max_mid_band5b_1: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
        max_mid_band5b_2: (
            Pipelines.Ingest,
            Pipelines.RCAL,
            Pipelines.FastImg,
            Pipelines.ICAL,
            Pipelines.DPrepA,
            Pipelines.DPrepA_Image,
            Pipelines.DPrepB,
            Pipelines.DPrepC,
            Pipelines.DPrepD,
        ),
    }

    all_hpsos = {
        hpso01,
        hpso02a,
        hpso02b,
        hpso04a,
        hpso04b,
        hpso04c,
        hpso05a,
        hpso05b,
        hpso13,
        hpso14,
        hpso15,
        hpso18,
        hpso22,
        hpso27and33,
        hpso32,
        hpso37a,
        hpso37b,
        hpso37c,
        hpso38a,
        hpso38b,
    }
    all_maxcases = {
        max_low,
        max_mid_band1,
        max_mid_band2,
        max_mid_band5a_1,
        max_mid_band5a_2,
        max_mid_band5b_1,
        max_mid_band5b_2,
    }
    available_hpsos = all_hpsos | all_maxcases


def define_symbolic_variables(o):
    """
    This method defines the *symbolic* variables that we will use during computations
    and that need to be kept symbolic during evaluation of formulae. One reason to do this would be to allow
    the output formula to be optimized by varying this variable (such as with Tsnap and Nfacet)

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.Nfacet = symbols(
        "Nfacet", integer=True, positive=True
    )  # Number of facets
    o.DeltaW_stack = symbols("DeltaW_stack", positive=True)

    return o


def define_design_equations_variables(o):
    """
    This method defines the *symbolic* variables that we will use during computations
    and that may need to be kept symbolic during evaluation. One reason to do this would be to allow
    the output formula to be optimized by varying these variables

    :param o: A supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)

    o.NfOut = symbols("N_f\\,out", integer=True, positive=True)
    o.Nmajor = symbols("N_major", integer=True, positive=True)
    o.Nfacet = symbols("N_facet", integer=True, positive=True)
    o.Ncu = symbols(
        "N_cu", integer=True, positive=True
    )  # number of compute units.
    o.RcuFLOP = symbols(
        "R_cu\\,FLOP", positive=True
    )  # peak FLOP capability of the compute unit
    o.RcuInter = symbols(
        "R_cu\\,inter", positive=True
    )  # maximum bandwidth of interconnect per Compute Unit
    o.RcuIo = symbols(
        "R_cu\\,io", positive=True
    )  # maximum I/O bandwidth of each compute unit to buffer
    o.Rfft = symbols("R_FFT", positive=True)
    o.Rinterfacet = symbols("R_interfacet", positive=True)
    o.Rrp = symbols("R_RP", positive=True)
    o.MuvGrid = symbols("M_uv\\,grid", positive=True)
    o.McuWork = symbols(
        "M_cu\\,work", positive=True
    )  # Size of main working memory of the compute unit
    o.McuPool = symbols(
        "M_cu\\,pool", positive=True
    )  # Size of slower (swap) working memory of the compute unit
    o.McuBuf = symbols(
        "M_cu\\,buf", positive=True
    )  # Size of buffer (or share of data-island local buffer)

    o.RspecFLOP = symbols("R^spec_FLOP", positive=True)
    o.RcontFLOP = symbols("R^cont_FLOP", positive=True)
    o.RfastFLOP = symbols("R^fast_FLOP", positive=True)

    o.RspecIo = symbols("R^spec_io", positive=True)
    o.RcontIo = symbols("R^cont_io", positive=True)

    o.Fci = symbols("F_ci", positive=True)
    o.MspecBufVis = symbols("M^spec_buf\\,vis", positive=True)
    o.McontBufVis = symbols("M^cont_buf\\,vis", positive=True)
    return o


def apply_global_parameters(o):
    """
    Applies the global parameters to the parameter container object o.

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.c = 299792458  # The speed of light, in m/s (from sympy.physics.units.c)
    o.Omega_E = (
        7.292115e-5  # Rotation relative to the fixed stars in radians/second
    )
    o.R_Earth = 6378136  # Radius if the Earth in meters (equal to astropy.const.R_earth.value)
    o.epsilon_w = 0.01  # Amplitude level of w-kernels to include
    # o.Mvis = 10.0  # Memory size of a single visibility datum in bytes. Set at 10 on 26 Jan 2016 (Ferdl Graser, CSP ICD)
    o.Mvis = 12  # Memory size of a single visibility datum in bytes. See below. Estimated value may change (again).
    o.Mjones = 64.0  # Memory size of a Jones matrix (taken from Ronald's calibration calculations)
    o.Naa = 9  # Support Size of the A Kernel, in (linear) Pixels.
    o.Nmm = 4  # Mueller matrix Factor: 1 is for diagonal terms only, 4 includes off-diagonal terms too.
    o.Npp = 4  # Number of polarization products
    o.Nw = 2  # Bytes per value
    o.Mpx = 8.0  # Memory size of an image pixel in bytes
    o.Mpx_out = 4.0  # Memory size of an output image pixel in bytes
    o.Mcpx = 16.0  # Memory size of a complex grid pixel in bytes
    o.NAteam = 10  # Number of A-team sources used in demixing
    # o.Qbw = 4.3 #changed from 1 to give 0.34 uv cells as the bw smearing limit. Should be investigated and linked to depend on amp_f_max, or grid_cell_error
    o.Qfcv = 1.0  # changed to 1 to disable but retain ability to see affect in parameter sweep.
    o.Qgcf = 8.0
    o.Qkernel = 10.0  #  epsilon_f/ o.Qkernel is the fraction of a uv cell we allow frequence smearing at edge of convoluion kernel to - i.e error on u,v, position one kernel-radius from gridding point.
    # o.grid_cell_error = 0.34 #found from tump time as given by SKAO at largest FoV (continuum).
    o.Qw = 1.0
    o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
    o.Qfov = 1.0  # Define this in case not defined below
    o.Qfov_nosmear = (
        0.67  # Field of view factor for visibility averaging calculation
    )
    o.amp_f_max = 1.02  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
    o.Tion = 10.0  # This was previously set to 60s (for PDR) May wish to use much smaller value.
    o.Nf_min = 40  # minimum number of channels to still enable distributed computing, and to reconstruct 5 Taylor terms
    o.FastImg_channels = 40  # minimum number of channels to still enable distributed computing, and to calculate spectral images
    o.Nf_min_gran = 800  # minimum number of channels in predict output to prevent overly large output sizes
    o.Ntt = 5  # Number of Taylor terms to compute
    o.NB_parameters = 500  # Number of terms in B parametrization
    o.r_facet_base = 0.2  # fraction of overlap (linear) in adjacent facets.
    o.max_subband_freq_ratio = 1.35  # maximum frequency ratio supported within each subband. 1.35 comes from Jeff Wagg SKAO ("30% fractional bandwidth in subbands").
    o.buffer_factor = 1  # The factor by which the buffer will be oversized. Factor 2 = "double buffering".
    o.Qfov_ICAL = 2.7  # Put the Qfov factor for the ICAL pipeline in here. It is used to calculate the correlator dump rate for instances where the maximum baseline used for an experiment is smaller than the maximum possible for the array. In that case, we might be able to request a longer correlator integration time in the correlator.
    o.Qmax_wproject = 1  # Maximum w-distance to use w-projection on (use w-stacking otherwise)

    # From CSP we are ingesting 10 bytes per visibility (single
    # polarization) built up as follows:
    #
    #   1 byte for time centroid
    # + 8 bytes for complex visibility
    # + 1 byte for flagging fraction, as per CSP ICD v.1.0.
    #     (i.e. 8 bytes for a complex value)
    # + 2 extra bytes for us to reconstruct timestamps etc.
    # + 2 bytes added by "ingest"
    # -----
    #  12 bytes per datum.
    #
    # Somewhere in the "Receive Visibilities" Function of the
    # Ingest Pipeline the 2 bytes additional information are
    # dropped (or used).  However, now we add:
    #
    # +4 byte for Weights (1 float)
    # +1 byte for Flags (1 bit per sample minimum).
    #
    # So the 12 bytes is the 8 for the complex value + 4 for the
    # Weight, which makes 12.  We should also add at least 1 bit
    # for Flags + some small amount of overhead for other meta
    # data.

    o.Nsource_find_iterations = 10  # Number of iterations in source finding
    o.Nsource = (
        1000  # Number of point sources modelled TODO: Should be set per HPSO
    )
    o.Nminor = 1000  # Average number of minor cycles per major cycle
    o.Nsolve = 10  # Number of Stefcal iterations
    o.Nscales = 10  # Number of scales in MS-MFS - updated to match requirement: SDP_REQ-676 (L2)
    o.Npatch = 4097  # Number of pixels in clean patch

    # To be overridden by the pipelines
    o.Nmajor = 2
    o.Nselfcal = 3
    o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1)
    o.NAProducts = "all"  # Number of A^A terms to be modelled
    o.tRCAL_G = 10.0  # Real time solution interval for Antenna gains
    o.tICAL_G = 1.0  # Solution interval for Antenna gains
    o.tICAL_B = 3600.0  # Solution interval for Bandpass
    o.tICAL_I = 10.0  # Solution interval for Ionosphere
    o.NIpatches = 1  # Number of ionospheric patches to solve
    o.Tsolve = (
        10 * 60
    )  # Calibration solution process frequency (task granularity)
    o.Tsnap_min = 0.1
    o.Tsnap = 10 * 60

    # Pipeline variants
    o.on_the_fly = False
    o.blcoal = True
    o.global_blcoal = False
    o.scale_predict_by_facet = True
    o.image_gridding = 0  # Pixels to pad to allow image-plane gridding

    return o


def apply_telescope_parameters(o, telescope):
    """
    Applies the parameters that apply to the supplied telescope to the parameter container object o

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :param telescope:
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    if not hasattr(o, "telescope") or o.telescope != telescope:
        o.set_param("telescope", telescope)

    if telescope.startswith(Telescopes.SKA1_Low):
        # Effective station diameter defined to be 38 metres in ECP-170049.
        o.Ds = 38  # station diameter in metres
        if telescope == Telescopes.SKA1_Low:
            o.Bmax = 74000  # Actually constructed max baseline in *m*
            o.Na = 512  # number of stations
            o.Nf_max = 65536  # maximum number of channels
            # Baseline length distribution calculated from layout in
            # SKA-TEL-SKO-0000422, Rev 03 (corresponding to ECP-170049),
            # see Absolute_Baseline_length_distribution.ipynb
            o.baseline_bins = np.array(
                (
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                (
                    47.90698386,
                    14.21844423,
                    16.78923068,
                    14.42789873,
                    6.65744251,
                )
            )
        elif telescope == Telescopes.SKA1_Low_AA05:
            o.Bmax = 125  # Actually constructed max baseline in *m*
            o.Na = 6  # number of stations
            o.Nf_max = 65536 // 4  # maximum number of channels
            o.baseline_bins = np.array(
                (
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                [0.0, 0.0, 0.0, 46.66666667, 53.33333333]
            )
        elif telescope == Telescopes.SKA1_Low_AA1:
            o.Bmax = 5850  # Actually constructed max baseline in *m*
            o.Na = 18  # number of stations
            o.Nf_max = 65536 // 4  # maximum number of channels
            o.baseline_bins = np.array(
                (
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                [29.41176471, 0.0, 0.0, 0.0, 70.58823529]
            )
        elif telescope == Telescopes.SKA1_Low_AA2:
            o.Bmax = 40000  # Actually constructed max baseline in *m*
            o.Na = 64  # number of stations
            o.Nf_max = 27778  # maximum number of channels
            o.baseline_bins = np.array(
                (
                    o.Bmax / 64.0,
                    o.Bmax / 32.0,
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                [
                    38.54166667,
                    3.125,
                    0.0,
                    12.45039683,
                    28.62103175,
                    0.0,
                    17.26190476,
                ]
            )
        elif telescope == Telescopes.SKA1_Low_AA15p:
            o.Bmax = 65000  # Actually constructed max baseline in *m*
            o.Na = 40  # number of stations
            o.Nf_max = 13824  # maximum number of channels
            o.baseline_bins = np.array(
                (
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                [7.43589744, 3.33333333, 25.76923077, 33.07692308, 30.38461538]
            )
        elif telescope == Telescopes.SKA1_Low_AA2p:
            o.Bmax = 65000  # Actually constructed max baseline in *m*
            o.Na = 68  # number of stations
            o.Nf_max = 27778  # maximum number of channels
            o.baseline_bins = np.array(
                (
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                [10.456942, 4.87697715, 30.62390158, 30.71177504, 23.33040422]
            )
        elif telescope == Telescopes.SKA1_Low_AA3:
            o.Bmax = 74000  # Actually constructed max baseline in *m*
            o.Na = 306  # number of stations
            o.Nf_max = 55556  # maximum number of channels
            o.baseline_bins = np.array(
                (
                    o.Bmax / 16.0,
                    o.Bmax / 8.0,
                    o.Bmax / 4.0,
                    o.Bmax / 2.0,
                    o.Bmax,
                )
            )
            o.baseline_bin_distribution = np.array(
                [
                    44.09514626,
                    22.88224579,
                    10.95253402,
                    11.34683382,
                    10.72324012,
                ]
            )
        else:
            raise Exception("Unknown Low telescope!")
        o.B_dump_ref = o.Bmax  # m
        o.Nbeam = 1  # number of beams
        o.Tint_min = 0.84934656  # Minimum correlator integration time (dump time) in *sec*
        # o.amp_f_max = 1.08  # Added by Rosie Bolton, 1.02 is consistent with the dump time of 0.08s at 200km BL.
        # o.NAProducts = o.nr_baselines # We must model the ionosphere for each station
        o.NAProducts = "all"  # We must model the ionosphere for each station
        o.tRCAL_G = 10.0
        o.tICAL_G = 1.0  # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 10.0  # Solution interval for Ionosphere
        o.NIpatches = 30  # Number of ionospheric patches to solve

    elif telescope == Telescopes.SKA1_Mid:
        o.Bmax = 150000  # Actually constructed max baseline in *m*
        o.Ds = 13.5  # dish diameter in metres, assume 13.5 as this matches the MeerKAT dishes
        o.Na = (
            64 + 133
        )  # number of dishes (expressed as the sum of MeerKAT and new dishes)
        o.Nbeam = 1  # number of beams
        o.Nf_max = 65536  # maximum number of channels
        o.Tint_min = 0.14  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.B_dump_ref = 150000  # m
        # Baseline length distribution calculated from layout in
        # SKA-TEL-INSA-0000537, Rev 04 (corresponding to ECP-1800002),
        # see Absolute_Baseline_length_distribution.ipynb
        o.baseline_bins = np.array(
            (
                5000.0,
                7500.0,
                10000.0,
                15000.0,
                25000.0,
                35000.0,
                55000.0,
                75000.0,
                90000.0,
                110000.0,
                130000.0,
                150000.0,
            )
        )
        o.baseline_bin_distribution = np.array(
            (
                6.13646961e01,
                5.16553546e00,
                2.87031760e00,
                4.98937879e00,
                6.32609709e00,
                4.63706544e00,
                5.73545412e00,
                5.50230558e00,
                1.80301539e00,
                1.45070204e00,
                1.08802653e-01,
                4.66297083e-02,
            )
        )
        # o.NAProducts = 3 # Most antennas can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 10.0
        o.tICAL_G = 1.0  # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 10.0  # Solution interval for Ionosphere
        o.NIpatches = 1  # Number of ionospheric patches to solve
        # o.Tion = 3600

    elif telescope == Telescopes.SKA1_Mid_AA2:
        o.Bmax = 110000  # Actually constructed max baseline in *m*
        o.Ds = 13.5  # dish diameter in metres, assume 13.5 as this matches the MeerKAT dishes
        o.Na = 64  # number of dishes (expressed as the sum of MeerKAT and new dishes)
        o.Nbeam = 1  # number of beams
        o.Nf_max = 65536 // 4  # maximum number of channels
        o.Tint_min = 0.19  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.B_dump_ref = 110000  # m
        # Baseline length distribution calculated from layout in
        # SKA-TEL-INSA-0000537, Rev 04 (corresponding to ECP-1800002),
        # see Absolute_Baseline_length_distribution.ipynb
        o.baseline_bins = np.array(
            (
                o.Bmax / 30.0,
                o.Bmax / 20.0,
                o.Bmax / 15.0,
                o.Bmax / 10.0,
                o.Bmax / 6.0,
                o.Bmax / 3.0,
                o.Bmax / 2.0,
                2 * o.Bmax / 3.0,
                o.Bmax,
            )
        )
        o.baseline_bin_distribution = np.array(
            [
                63.14484127,
                1.19047619,
                1.68650794,
                4.41468254,
                13.88888889,
                9.72222222,
                2.7281746,
                0.09920635,
                3.125,
            ]
        )
        # o.NAProducts = 3 # Most antennas can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 10.0
        o.tICAL_G = 1.0  # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 10.0  # Solution interval for Ionosphere
        o.NIpatches = 1  # Number of ionospheric patches to solve
        # o.Tion = 3600

    elif telescope == Telescopes.SKA1_Mid_AAs:
        o.Bmax = 160000  # Actually constructed max baseline in *m*
        o.Ds = 13.5  # dish diameter in metres, assume 13.5 as this matches the MeerKAT dishes
        o.Na = 144  # number of dishes (expressed as the sum of MeerKAT and new dishes)
        o.Nbeam = 1  # number of beams
        o.Nf_max = 65536  # maximum number of channels
        o.Tint_min = 0.19  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.B_dump_ref = 160000  # m
        # Baseline length distribution calculated from layout in
        # SKA-TEL-INSA-0000537, Rev 04 (corresponding to ECP-1800002),
        # see Absolute_Baseline_length_distribution.ipynb
        o.baseline_bins = np.array(
            (
                5000.0,
                7500.0,
                10000.0,
                15000.0,
                25000.0,
                35000.0,
                55000.0,
                75000.0,
                90000.0,
                110000.0,
                130000.0,
                160000.0,
            )
        )
        o.baseline_bin_distribution = np.array(
            [
                46.60514924,
                6.33401686,
                3.70243791,
                6.70995671,
                8.80610617,
                6.49350649,
                8.24789246,
                7.79220779,
                2.75689223,
                2.15311005,
                0.23923445,
                0.15948963,
            ]
        )
        # o.NAProducts = 3 # Most antennas can be modelled as the same. [deactivated for now]
        o.tRCAL_G = 10.0
        o.tICAL_G = 1.0  # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 10.0  # Solution interval for Ionosphere
        o.NIpatches = 1  # Number of ionospheric patches to solve
        # o.Tion = 3600

    elif telescope == Telescopes.LOFAR_HBA:
        o.Ds = 41.05  # station diameter in metres
        o.Bmax = 121000  # Actually constructed max baseline in *m*
        o.Na = 62  # number of stations
        o.Nf_max = 488 * 256  # maximum number of channels
        o.baseline_bins = np.array(
            (
                o.Bmax / 64.0,
                o.Bmax / 32.0,
                o.Bmax / 16.0,
                o.Bmax / 8.0,
                o.Bmax / 4.0,
                o.Bmax / 2.0,
                o.Bmax,
            )
        )
        o.baseline_bin_distribution = np.array(
            (
                52.2474881,
                9.36012692,
                5.39397144,
                8.67265997,
                8.93707033,
                11.95134849,
                3.43733474,
            )
        )
        o.B_dump_ref = o.Bmax  # m
        o.Nbeam = 1  # number of beams
        o.Tint_min = 1.0006633  # Minimum correlator integration time (dump time) in *sec* - in reference design
        o.NAProducts = "all"  # We must model the ionosphere for each station
        o.tRCAL_G = 10.0
        o.tICAL_G = 1.0  # Solution interval for Antenna gains
        o.tICAL_B = 3600.0  # Solution interval for Bandpass
        o.tICAL_I = 10.0  # Solution interval for Ionosphere
        o.NIpatches = 30  # Number of ionospheric patches to solve

    else:
        raise Exception(f"Unknown Telescope: {telescope}")

    o.telescope = telescope
    return o


def apply_band_parameters(o, band):
    """
    Applies the parameters that apply to the band to the parameter container object o

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :param band:
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    o.band = band
    if band == Bands.Low:
        o.freq_min = 0.05e9
        o.freq_max = 0.35e9
    elif band == Bands.LowAA1:
        o.freq_min = 0.075e9
        o.freq_max = 0.175e9
    elif band == Bands.LowAA2:
        o.freq_min = 0.075e9
        o.freq_max = 0.225e9
    elif band == Bands.Mid1:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 0.35e9
        o.freq_max = 1.05e9
    elif band == Bands.Mid2:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 0.95e9
        o.freq_max = 1.76e9
    elif band == Bands.Mid5a:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 4.6e9
        o.freq_max = 8.5e9
    elif band == Bands.Mid5b:
        o.telescope = Telescopes.SKA1_Mid
        o.freq_min = 8.3e9
        o.freq_max = 15.4e9
    # Taken from https://www.aanda.org/articles/aa/pdf/2013/08/aa20873-12.pdf table 2
    elif band == Bands.LofarHigh1:  # 200 MHz sampling (2nd Nyquist zone)
        o.telescope = Telescopes.LOFAR_HBA
        o.freq_min = 110e6
        o.freq_max = 190e6
    elif band == Bands.LofarHigh2:  # 160 MHz sampling (3rd Nyquist zone)
        o.telescope = Telescopes.LOFAR_HBA
        o.freq_min = 170e6
        o.freq_max = 230e6
    elif band == Bands.LofarHigh3:  # 200 MHz sampling (3rd Nyquist zone)
        o.telescope = Telescopes.LOFAR_HBA
        o.freq_min = 210e6
        o.freq_max = 220e6
    else:
        raise Exception(f"Unknown Band: {band}")

    return o


def define_pipeline_products(o, pipeline, named_pipeline_products=[]):
    o.pipeline = pipeline
    o.products = {}
    for product in named_pipeline_products:
        o.products[product] = {
            "Rflop": 0,
            "Rio": 0.0,
            "Rinteract": 0.0,
            "MW_cache": 0,
        }
    return o


def apply_pipeline_parameters(o, pipeline):
    """
    Applies the parameters that apply to the pipeline to the parameter container object o

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :param pipeline: Type of pipeline
    :raise Exception:
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    define_pipeline_products(o, pipeline)

    if pipeline == Pipelines.Ingest:
        o.Nf_out = o.Nf_max
        o.Nselfcal = 0
        o.Nmajor = 0
        o.Nmajortotal = 0
        o.Npp = 4  # We get everything
        o.Tobs = 1.0 * 3600.0  # in seconds
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.08
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.ICAL:
        o.Qfov = o.Qfov_ICAL  # Field of view factor
        o.Nselfcal = 3
        o.Nmajor = 2
        o.Nminor = 10000
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Npp = 4  # We get everything
        o.Tobs = 1.0 * 3600.0  # in seconds
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.08
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.RCAL:
        o.Qfov = 2.7  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 0
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Npp = 4  # We get everything
        o.Tobs = 1.0 * 3600.0  # in seconds
        o.Tsolve = 10
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.08
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.034

    elif (pipeline == Pipelines.DPrepA) or (
        pipeline == Pipelines.DPrepA_Image
    ):
        o.Qfov = 1.0  # Field of view factor
        o.Qfov_out = 0.42  # Output image FoV (to half-power point)
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Npp = 2  # We only want Stokes I, V
        o.Tobs = 1.0 * 3600.0  # in seconds
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.08
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.DPrepB:
        o.Qfov = 1.0  # Field of view factor
        o.Qfov_out = 0.42  # Output image FoV (to half-power point)
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(o.Nf_min, o.Nf_max)
        o.Npp = 4  # We want Stokes I, Q, U, V
        o.Tobs = 1.0 * 3600.0  # in seconds
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.08
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.034

    elif pipeline == Pipelines.DPrepC:
        o.Qfov = 1.0  # Field of view factor
        o.Qfov_out = 0.42  # Output image FoV (to half-power point)
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = o.Nf_max  # The same as the maximum number of channels
        o.Npp = 4  # We want Stokes I, Q, U, V
        o.Tobs = 1.0 * 3600
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.02
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.01

    elif pipeline == Pipelines.DPrepD:
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 10
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 2.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = o.Nf_max  # The same as the maximum number of channels
        o.Npp = 4  # We want Stokes I, Q, U, V
        o.Tint_out = o.Tint_min  # Integration time for averaged visibilities
        o.Tobs = 1.0 * 3600
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.02
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.01

    elif pipeline == Pipelines.FastImg:
        # TODO: check whether Naa (A kernel size) should be smaller (or zero) for fast imaging
        # TODO: update this to cope with multiple timescales for output
        o.Qfov = 1.0  # Field of view factor
        o.Nselfcal = 0
        o.Nmajor = 0
        o.Nmajortotal = o.Nmajor * (o.Nselfcal + 1) + 1
        o.Qpix = 1.5  # Quality factor of synthesised beam oversampling
        o.Nf_out = min(
            o.FastImg_channels, o.Nf_max
        )  # Initially this value was computed, but now capped to 500.
        o.Npp = 2  # We only want Stokes I, V
        o.Tobs = 1.0
        o.Tsnap = o.Tobs
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.amp_f_max = 1.02
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.amp_f_max = 1.02

        o.Nmm = 1  # Off diagonal terms probably not needed?

    elif pipeline == Pipelines.PSS:
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.Ntiedbeam = 500
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.Ntiedbeam = 1500
        else:
            o.Ntiedbeam = 0
        o.Nf_out = 128
        o.Tobs = 600

    elif pipeline == Pipelines.SinglePulse:
        if o.telescope.startswith(Telescopes.SKA1_Low):
            o.Ntiedbeam = 500
        elif o.telescope.startswith(Telescopes.SKA1_Mid):
            o.Ntiedbeam = 1500
        else:
            o.Ntiedbeam = 0
        o.Nf_out = 1024
        o.Npp = 4
        o.Tobs = 600

    elif pipeline == Pipelines.PST:
        o.Ntiedbeam = 16
        o.Nf_out = 4096
        o.Npp = 4
        o.Tobs = 1800

    else:
        raise Exception("Unknown pipeline: %s" % str(pipeline))

    return o


def apply_hpso_parameters(o, hpso, hpso_pipe):
    """
    Applies the parameters for the HPSO pipeline to the parameter container object o.

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :param hpso: The HPSO whose parameters we are applying
    :param hpso_pipe: The pipeline whose parameters we are applying
    :returns: ParameterContainer
    """
    assert isinstance(o, ParameterContainer)
    # Check that the HPSO is listed as being available for calculation
    assert hpso in HPSOs.available_hpsos
    # Check that the telescope lookup has been defined
    assert hpso in HPSOs.hpso_telescopes
    # Check that the pipeline lookup has been defined
    assert hpso in HPSOs.hpso_pipelines
    # Check that a valid pipeline has been defined for this HPSO
    assert hpso_pipe in HPSOs.hpso_pipelines[hpso]

    if not hasattr(o, "hpso") or o.hpso != hpso:
        o.set_param("hpso", hpso)
    o.telescope = HPSOs.hpso_telescopes[hpso]
    o.pipeline = hpso_pipe

    if hpso == HPSOs.max_low:
        # Maximal case for Low.

        o.band = Bands.Low
        o.freq_min = 0.05e9
        o.freq_max = 0.35e9
        o.Nbeam = 1
        o.Nf_max = 65536
        o.Bmax = 65000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 9.0

    elif hpso == HPSOs.max_mid_band1:
        # Maximal case for Mid band 1.

        o.band = Bands.Mid1
        o.freq_min = 0.35e9
        o.freq_max = 1.05e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 1.4

    elif hpso == HPSOs.max_mid_band2:
        # Maximal case for Mid band 2.

        o.band = Bands.Mid2
        o.freq_min = 0.95e9
        o.freq_max = 1.76e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 1.4

    elif hpso == HPSOs.max_mid_band5a_1:
        # Maximal case for Mid band 5a, lowest 2.5 GHz.

        o.band = Bands.Mid5a
        o.freq_min = 4.6e9
        o.freq_max = 7.1e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 1.4

    elif hpso == HPSOs.max_mid_band5a_2:
        # Maximal case for Mid band 5a, highest 2.5 GHz.

        o.band = Bands.Mid5a
        o.freq_min = 6.0e9
        o.freq_max = 8.5e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 1.4

    elif hpso == HPSOs.max_mid_band5b_1:
        # Maximal case for Mid band 5b, lowest 2.5 GHz.

        o.band = Bands.Mid5b
        o.freq_min = 8.3e9
        o.freq_max = 10.8e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 1.4

    elif hpso == HPSOs.max_mid_band5b_2:
        # Maximal case for Mid band 5b, highest 2.5 GHz.

        o.band = Bands.Mid5b
        o.freq_min = 12.9e9
        o.freq_max = 15.4e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 6.0 * 3600.0
        o.Tpoint = 6.0 * 3600.0
        o.Texp = 6.0 * 3600.0

        if hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 65536
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 1.4

    elif hpso == HPSOs.hpso01:
        # EoR - Imaging.

        o.band = Bands.Low
        o.freq_min = 0.05e9
        o.freq_max = 0.20e9
        o.Nbeam = 2
        o.Nf_max = 65536 // o.Nbeam
        o.Bmax = 65000
        o.Tobs = 5 * 3600.0
        o.Tpoint = 2000 * 3600.0
        o.Texp = 5000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 2.7
        elif hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
            o.Npp = 4
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 1500
            o.Npp = 4
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 9.0
            o.Npp = 4

    elif hpso == HPSOs.hpso02a:
        # EoR - Power Spectrum.

        o.band = Bands.Low
        o.freq_min = 0.05e9
        o.freq_max = 0.20e9
        o.Nbeam = 2
        o.Nf_max = 65536 // o.Nbeam
        o.Bmax = 65000
        o.Tobs = 5 * 3600.0
        o.Tpoint = 200 * 3600.0
        o.Texp = 5000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.8
        elif hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
            o.Npp = 4
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 1500
            o.Npp = 4
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 9.0
            o.Npp = 4

    elif hpso == HPSOs.hpso02b:
        # EoR - Power Spectrum.

        o.band = Bands.Low
        o.freq_min = 0.05e9
        o.freq_max = 0.20e9
        o.Nbeam = 2
        o.Nf_max = 65536 // o.Nbeam
        o.Bmax = 65000
        o.Tobs = 5 * 3600.0
        o.Tpoint = 20 * 3600.0
        o.Texp = 5000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.8
        elif hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 500
            o.Npp = 4
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 1500
            o.Npp = 4
        elif hpso_pipe == Pipelines.DPrepD:
            o.Nf_out = o.Nf_max // 4
            o.Tint_out = 9.0
            o.Npp = 4

    elif hpso == HPSOs.hpso04a:
        # Pulsar Search.

        o.band = Bands.Low
        o.freq_min = 0.15e9
        o.freq_max = 0.35e9
        o.Nbeam = 1
        o.Nf_max = 65536
        o.Bmax = 65000
        o.Tobs = 40 * 60.0
        o.Tpoint = 40 * 60.0
        o.Texp = 12750 * 3600.0

    elif hpso == HPSOs.hpso04b:
        # Pulsar Search.

        o.band = Bands.Mid1
        o.freq_min = 0.65e9
        o.freq_max = 0.95e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 10 * 60.0
        o.Tpoint = 10 * 60.0
        o.Texp = 800 * 3600.0

    elif hpso == HPSOs.hpso04c:
        # Pulsar Search.

        o.band = Bands.Mid2
        o.freq_min = 1.25e9
        o.freq_max = 1.55e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 10 * 60.0
        o.Tpoint = 10 * 60.0
        o.Texp = 2400 * 3600.0

    elif hpso == HPSOs.hpso05a:
        # Pulsar Timing.

        o.band = Bands.Low
        o.freq_min = 0.15e9
        o.freq_max = 0.35e9
        o.Nbeam = 1
        o.Nf_max = 65536
        o.Bmax = 65000
        o.Tobs = 40 * 60.0
        o.Tpoint = 40 * 60.0
        o.Texp = 4300 * 3600.0

    elif hpso == HPSOs.hpso05b:
        # Pulsar Timing.

        o.band = Bands.Mid2
        o.freq_min = 0.95e9
        o.freq_max = 1.76e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 15 * 60.0
        o.Tpoint = 15 * 60.0
        o.Texp = 1600 * 3600.0

    elif hpso == HPSOs.hpso13:
        # HI - High z.

        o.band = Bands.Mid1
        o.freq_min = 0.79e9
        o.freq_max = 0.95e9
        # 40k comes from assuming 4 kHz width over 790-950 MHz.
        o.Nf_max = 40000
        o.Bmax = 35000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 1000 * 3600.0
        o.Texp = 5000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 2.7
        elif hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 160
            o.Npp = 2
        elif hpso_pipe == Pipelines.DPrepC:
            o.Nf_out = 3200
            o.Npp = 2

    elif hpso == HPSOs.hpso14:
        # HI - Low z.

        o.band = Bands.Mid2
        # Increase freq range to give > 1.2 ratio for continuum.
        o.freq_min = 1.2e9
        o.freq_max = 1.5e9
        o.Nf_max = 65536
        # Max baseline of 25 km set by experiment.
        o.Bmax = 25000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 200 * 3600.0
        o.Texp = 2000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.8
        elif hpso_pipe == Pipelines.DPrepB:
            # 300 channel pseudo-continuum (small BW)
            o.Nf_out = 300
            o.Npp = 2
        elif hpso_pipe == Pipelines.DPrepC:
            o.freq_min = 1.3e9
            o.freq_max = 1.4e9
            # Only 5,000 spectral line channels.
            o.Nf_out = 5000
            o.Npp = 2

    elif hpso == HPSOs.hpso15:
        # HI - Galaxy.

        o.band = Bands.Mid2
        # Requested freq range is 1415 - 1425 MHz, increased to give
        # larger frac BW in continuum.
        o.freq_min = 1.30e9
        o.freq_max = 1.56e9
        o.Nf_max = 65536
        o.Bmax = 15000
        o.Tobs = 4.4 * 3600.0
        o.Tpoint = 4.4 * 3600.0
        o.Texp = 12600 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.8
        elif hpso_pipe == Pipelines.DPrepB:
            # 300 channels pseudo-continuum.
            o.Nf_out = 260
            o.Npp = 2
        elif hpso_pipe == Pipelines.DPrepC:
            o.freq_min = 1.415e9
            o.freq_max = 1.425e9
            # Only 2,500 spectral line channels.
            o.Nf_out = 2500
            o.Npp = 2

    elif hpso == HPSOs.hpso18:
        # Transients - FRB.

        o.band = Bands.Mid1
        o.freq_min = 0.65e9
        o.freq_max = 0.95e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 1 * 60.0
        o.Tpoint = 1 * 60.0
        o.Texp = 10000 * 3600.0

    elif hpso == HPSOs.hpso22:
        # CoL - Planet Formation.

        o.band = Bands.Mid5b
        o.freq_min = 8.0e9
        o.freq_max = 12.0e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 600 * 3600.0
        o.Texp = 6000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 2.7
        elif hpso_pipe == Pipelines.DPrepB:
            # 500-channel continuum observation.
            o.Nf_out = 1000
            o.Npp = 4

    elif hpso == HPSOs.hpso27and33:
        # Combination of two HPSOs done commensally.
        # 27: Magnetism - RM Grid.
        # 33: Cosmology - ISW, Dipole

        o.band = Bands.Mid2
        o.freq_min = 1.0e9
        o.freq_max = 1.7e9
        o.Nf_max = 65536
        o.Bmax = 30000
        o.Tobs = 7.4 * 60.0
        o.Tpoint = 7.4 * 60.0
        o.Texp = 10000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.0
        elif hpso_pipe == Pipelines.DPrepB:
            # Continuum experiment with 500 output channels.
            o.Nf_out = 700
            o.Npp = 4

    elif hpso == HPSOs.hpso32:
        # Cosmology - High z IM.

        # This HPSO uses auto-correlation observations. Requirements
        # for interferometric imaging to support it are under
        # development.

        o.band = Bands.Mid1
        o.freq_min = 0.35e9
        o.freq_max = 1.05e9
        o.Nf_max = 65536
        o.Bmax = 20000
        o.Tobs = 2.2 * 3600.0
        o.Tpoint = 2.2 * 3600.0
        o.Texp = 10000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.0
        elif hpso_pipe == Pipelines.DPrepB:
            # 700 channels required in output continuum cubes.
            o.Nf_out = 700
            o.Npp = 2

    elif hpso == HPSOs.hpso37a:
        # Continuum - SFR(z).

        o.band = Bands.Mid2
        o.freq_min = 1.0e9
        o.freq_max = 1.7e9
        o.Nf_max = 65536
        o.Bmax = 120000
        o.Tobs = 3.8 * 3600.0
        o.Tpoint = 3.8 * 3600.0
        o.Texp = 10000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.8
        elif hpso_pipe == Pipelines.DPrepB:
            # 700 channels required in output continuum cubes.
            o.Nf_out = 700
            o.Npp = 4

    elif hpso == HPSOs.hpso37b:
        # Continuum - SFR(z).

        o.band = Bands.Mid2
        o.freq_min = 1.0e9
        o.freq_max = 1.7e9
        o.Nf_max = 65536
        o.Bmax = 120000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 95 * 3600.0
        o.Texp = 2000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 2.7
        elif hpso_pipe == Pipelines.DPrepB:
            # 700 channels required in output continuum cubes.
            o.Nf_out = 700
            o.Npp = 4

    elif hpso == HPSOs.hpso37c:
        # Continuum - SFR(z).

        o.band = Bands.Mid2
        o.freq_min = 1.0e9
        o.freq_max = 1.7e9
        o.Nf_max = 65536
        o.Bmax = 120000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 2000 * 3600.0
        o.Texp = 2000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 2.7
        elif hpso_pipe == Pipelines.DPrepB:
            # 700 channels required in output continuum cubes
            o.Nf_out = 700
            o.Npp = 4

    elif hpso == HPSOs.hpso38a:
        # Continuum - SFR(z).

        o.band = Bands.Mid5a
        o.freq_min = 7.0e9
        o.freq_max = 11.0e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 16.4 * 3600.0
        o.Texp = 1000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 1.8
        elif hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 1000
            o.Npp = 4

    elif hpso == HPSOs.hpso38b:
        # Continuum - SFR(z).

        o.bands = Bands.Mid5a
        o.freq_min = 7.0e9
        o.freq_max = 11.0e9
        o.Nf_max = 65536
        o.Bmax = 150000
        o.Tobs = 8 * 3600.0
        o.Tpoint = 1000 * 3600.0
        o.Texp = 1000 * 3600.0

        if hpso_pipe == Pipelines.ICAL:
            o.Qfov = 2.7
        elif hpso_pipe == Pipelines.DPrepB:
            o.Nf_out = 1000
            o.Npp = 4

    else:
        raise Exception("Unknown HPSO %s!" % hpso)

    return o


def apply_yaml_parameters(o, yaml_parameters):
    """
    Applies the custom parameters specified in a yaml file to the ParameterContainer object o.

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :param yaml_parameters: A dictionary of parameters (keys) and their values to be applied to the ParameterContainer object
    :returns: ParameterContainer
    """
    for key in yaml_parameters:
        if (key == "baseline_bin_distribution") or (key == "baseline_bins"):
            # Must convert Python list to a NumPy array, otherwise the scheduling and simulation code fails
            setattr(o, key, np.array(yaml_parameters[key]))
        else:
            setattr(o, key, yaml_parameters[key])
    return o


def apply_custom_array_parameters(
    o, array_config_file, array_config_bins, Bmax
):
    """
    Calculates the values of `baseline_bin_distribution` and `baseline_bins parameters`, from a custom set of baselines and applied them to the ParameterContainer object o.

    :param o: The supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
    :param yaml_parameters: A dictionary of parameters (keys) and their values to be applied to the ParameterContainer object
    :returns: ParameterContainer
    """

    row_dtypes = np.dtype([("name", "U10"), ("lon", "f8"), ("lat", "f8")])
    rows = np.loadtxt(array_config_file, usecols=(0, 1, 2), dtype=row_dtypes)
    assert (
        rows[0]["name"] == "Centre"
    ), "First entry in layout file must be the centre of the array"

    centre_antenna = rows[0]
    all_other_antennas = rows[1:]

    histogram_bin_edges = np.linspace(
        0, Bmax, array_config_bins + 1, dtype=np.float64
    )
    o.baseline_bins = np.array(histogram_bin_edges[1:])
    o.baseline_bin_distribution = antenna_positions_to_baseline_histogram(
        centre_antenna, all_other_antennas, histogram_bin_edges
    )

    return o


def antenna_positions_to_baseline_histogram(
    centre_antenna: np.void,
    all_other_antennas: np.ndarray,
    histogram_bin_edges: np.ndarray,
) -> np.ndarray:
    """Converts antenna positions to a histogram of baseline lengths.

    Args:
        centre_antenna (np.void): latitude and longitude of the centre antenna.
        all_other_antennas (np.ndarray): array of the latitude and longitude of all the other antennas.
        histogram_bin_edges (np.ndarray): the histogram bin edges of the baselines include 0 and the maximum.

    Returns:
        histogram (np.ndarray): array of baseline length histogram values.
    """
    radius_earth = 6371010
    degrees_to_radians = np.pi / 180.0

    array_x_centre = (
        radius_earth
        * np.cos(centre_antenna["lat"] * degrees_to_radians)
        * np.cos(centre_antenna["lon"] * degrees_to_radians)
    )
    array_y_centre = (
        radius_earth
        * np.cos(centre_antenna["lat"] * degrees_to_radians)
        * np.sin(centre_antenna["lon"] * degrees_to_radians)
    )
    array_z_centre = radius_earth * np.sin(
        centre_antenna["lat"] * degrees_to_radians
    )

    array_dx = []
    array_dy = []
    array_dz = []
    array_r = []

    n_positions = len(all_other_antennas)

    for i in range(n_positions):
        array_dx.append(
            radius_earth
            * np.cos(all_other_antennas["lat"][i] * degrees_to_radians)
            * np.cos(all_other_antennas["lon"][i] * degrees_to_radians)
            - array_x_centre
        )
        array_dy.append(
            radius_earth
            * np.cos(all_other_antennas["lat"][i] * degrees_to_radians)
            * np.sin(all_other_antennas["lon"][i] * degrees_to_radians)
            - array_y_centre
        )
        array_dz.append(
            radius_earth
            * np.sin(all_other_antennas["lat"][i] * degrees_to_radians)
            - array_z_centre
        )
        array_r.append(
            np.sqrt(array_dx[i] ** 2 + array_dy[i] ** 2 + array_dz[i] ** 2)
        )

    baseline_lengths = []
    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            baseline_length = np.sqrt(
                (array_dx[j] - array_dx[i]) ** 2
                + (array_dy[j] - array_dy[i]) ** 2
                + (array_dz[j] - array_dz[i]) ** 2
            )
            baseline_lengths.append(baseline_length)

    histogram = np.histogram(
        baseline_lengths, bins=histogram_bin_edges, density=False
    )[0].astype(np.float64)

    return histogram


ALL_PARAMETER_KEYS = [
    "name",
    "telescope",
    "band",
    "pipeline",
    "array_config_file",
    "array_config_bins",
    "freq_max",
    "freq_min",
    "Bmax",
    "B_dump_ref",
    "baseline_bins",
    "baseline_bin_distribution",
    "Ds",
    "Na",
    "Nf_max",
    "Tint_min",
    "Tobs",
    "Tpoint",
    "Texp",
    "Nbeam",
    "Tint_out",
    "tRCAL_G",
    "tICAL_G",
    "tICAL_B",
    "tICAL_I",
    "NIpatches",
    "Nf_out",
    "Npp",
    "Nselfcal",
    "Nmajor",
    "Nmajortotal",
    "Nminor",
    "Qpix",
    "Qfov",
    "amp_f_max",
    "Ntiedbeam",
    "on_the_fly",
    "blcoal",
    "global_blcoal",
    "scale_predict_by_facet",
    "image_gridding",
    "c",
    "Omega_E",
    "R_Earth",
    "epsilon_w",
    "Mvis",
    "Mjones",
    "NAProducts",
    "Naa",
    "Nmm",
    "Nw",
    "Mpx",
    "Mpx_out",
    "Mcpx",
    "NAteam",
    "Qfcv",
    "Qgcf",
    "Qkernel",
    "Qw",
    "Tion",
    "Nf_min",
    "FastImg_channels",
    "Nf_min_gran",
    "Ntt",
    "NB_parameters",
    "r_facet_base",
    "max_subband_freq_ratio",
    "buffer_factor",
    "Qfov_ICAL",
    "Qmax_wproject",
    "Nsource_find_iterations",
    "Nsource",
    "Nminor",
    "Nsolve",
    "Nscales",
    "Npatch",
    "Tsolve",
    "Tsnap_min",
    "Tsnap",
    "Texp",
    "Tpoint",
    "Qfov_nosmear",
    "Qfov_out",
]
