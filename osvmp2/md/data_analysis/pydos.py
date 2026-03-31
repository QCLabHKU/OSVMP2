import os, sys, warnings
import numpy as np
from scipy.fftpack import fft
from scipy.signal import convolve, gaussian
from pwtools import constants, _flib, num
#from pwtools import constants, num
from pwtools.verbose import verbose
from pwtools.signal import pad_zeros, welch, mirror
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

def read_data(fname, sel=None, Natom=None):
    with open(fname, 'r') as f:
        lines = f.readlines()
        Natom = int(lines[0])
        atom_list = []
        for l in lines[2:2+Natom]:
            atom_list.append(l.split()[0])

    atom_mass = {'H':1, 'B':5, 'C':6, 'N':7, 'O':8, 
                 'F':9, 'SI':14, 'P':15, 'S':16}
    mass_list = []
    for atom in atom_list:
        mass_list.append(atom_mass[atom.upper()])
        
    with open(fname, 'r') as fo:
        timestep = 0
        coords = []
        for idx, line in enumerate(fo):
            if idx == 0:
                sel = Natom = int(line)
            try:
                next(fo)
            except StopIteration:
                break
            for n in range(sel):
                line = next(fo)
                info = line.split()
                coords.append(info[1:])
            timestep += 1
        coords = np.asfarray(coords, dtype=np.float64).reshape(timestep,-1,3)
        print(timestep)
    return coords, np.asarray(mass_list)

def pyvacf(vel, m=None, method=3):
    """Reference implementation for calculating the VACF of velocities in 3d
    array `vel`. This is slow. Use for debugging only. For production, use
    pyvacf().

    Parameters
    ----------
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        | 1 : 3 loops
        | 2 : replace 1 inner loop
        | 3 : replace 2 inner loops

    Returns
    -------
    c : 1d array (nstep,)
        VACF
    """
    natoms = vel.shape[1]
    nstep = vel.shape[0]
    c = np.zeros((nstep,), dtype=float)
    if m is None:
        m = np.ones((natoms,), dtype=float)
    if method == 1:
        # c(t) = <v(t0) v(t0 + t)> / <v(t0)**2> = C(t) / C(0)
        #
        # "displacements" `t'
        for t in range(nstep):
            # time origins t0 == j
            for j in range(nstep-t):
                for i in range(natoms):
                    c[t] += np.dot(vel[j,i,:], vel[j+t,i,:]) * m[i]
    elif method == 2:
        # replace 1 inner loop
        for t in range(nstep):
            for j in range(nstep-t):
                # (natoms, 3) * (natoms, 1) -> (natoms, 3)
                c[t] += (vel[j,...] * vel[j+t,...] * m[:,None]).sum()
    elif method == 3:
        # replace 2 inner loops:
        # (xx, natoms, 3) * (1, natoms, 1) -> (xx, natoms, 3)
        for t in range(nstep):
            c[t] = (vel[:(nstep-t),...] * vel[t:,...]*m[None,:,None]).sum()
    else:
        raise ValueError('unknown method: %s' %method)
    # normalize to unity
    c = c / c[0]
    return c

def fvacf(vel, m=None, method=2, nthreads=None):
    """Interface to Fortran function _flib.vacf(). Otherwise same
    functionallity as pyvacf(). Use this for production calculations.

    Parameters
    ----------
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        | 1 : loops
        | 2 : vectorized loops

    nthreads : int ot None
        If int, then use this many OpenMP threads in the Fortran extension.
        Only useful if the extension was compiled with OpenMP support, of
        course.

    Returns
    -------
    c : 1d array (nstep,)
        VACF

    Notes
    -----
    Fortran extension::

        $ python -c "import _flib; print(_flib.vacf.__doc__)"
        vacf - Function signature:
          c = vacf(v,m,c,method,use_m,[nthreads,natoms,nstep])
        Required arguments:
          v : input rank-3 array('d') with bounds (natoms,3,nstep)
          m : input rank-1 array('d') with bounds (natoms)
          c : input rank-1 array('d') with bounds (nstep)
          method : input int
          use_m : input int
        Optional arguments:
          nthreads : input int
          natoms := shape(v,0) input int
          nstep := shape(v,2) input int
        Return objects:
          c : rank-1 array('d') with bounds (nstep)

    Shape of `vel`: The old array shapes were (natoms, 3, nstep), the new is
        (nstep,natoms,3). B/c we don't want to adapt flib.f90, we change
        vel's shape before passing it to the extension.

    See Also
    --------
    :mod:`pwtools._flib`
    :func:`vacf_pdos`
    """
    # f2py copies and C-order vs. Fortran-order arrays
    # ------------------------------------------------
    # With vel = np.asarray(vel, order='F'), we convert vel to F-order and a
    # copy is made by numpy. If we don't do it, the f2py wrapper code does.
    # This copy is unavoidable, unless we allocate the array vel in F-order in
    # the first place.
    #   c = _flib.vacf(np.asarray(vel, order='F'), m, c, method, use_m)
    #
    # speed
    # -----
    # The most costly step is calculating the VACF. FFTing that is only the fft
    # of a 1d-array which is fast, even if the length is not a power of two.
    # Padding is not needed.
    #
    natoms = vel.shape[1]
    nstep = vel.shape[0]
    assert vel.shape[-1] == 3, ("last dim of vel must be 3: (nstep,natoms,3)")
    # `c` as "intent(in, out)" could be "intent(out), allocatable" or so,
    # makes extension more pythonic, don't pass `c` in, let be allocated on
    # Fortran side
    c = np.zeros((nstep,), dtype=float)
    if m is None:
        # dummy
        m = np.empty((natoms,), dtype=float)
        use_m = 0
    else:
        use_m = 1
    verbose("calling _flib.vacf ...")
    if nthreads is None:
        # Possible f2py bug workaround: The f2py extension does not always set
        # the number of threads correctly according to OMP_NUM_THREADS. Catch
        # OMP_NUM_THREADS here and set number of threads using the "nthreads"
        # arg.
        key = 'OMP_NUM_THREADS'
        if key in os.environ:
            nthreads = int(os.environ[key])
            c = _flib.vacf(vel, m, c, method, use_m, nthreads)
        else:
            c = _flib.vacf(vel, m, c, method, use_m)
    else:
        c = _flib.vacf(vel, m, c, method, use_m, nthreads)
    verbose("... ready")
    return c

def pdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
         npad=None, tonext=False, mirr=False, method='direct'):
    """Phonon DOS by FFT of the VACF or direct FFT of atomic velocities.

    Integral area is normalized to `area`. It is possible (and recommended) to
    zero-padd the velocities (see `npad`).

    Parameters
    ----------
    vel : 3d array (nstep, natoms, 3)
        atomic velocities
    dt : time step
    m : 1d array (natoms,),
        atomic mass array, if None then mass=1.0 for all atoms is used
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool
        use Welch windowing on data before FFT (reduces leaking effect,
        recommended)
    npad : {None, int}
        method='direct' only: Length of zero padding along `axis`. `npad=None`
        = no padding, `npad > 0` = pad by a length of ``(nstep-1)*npad``. `npad
        > 5` usually results in sufficient interpolation.
    tonext : bool
        method='direct' only: Pad `vel` with zeros along `axis` up to the next
        power of two after the array length determined by `npad`. This gives
        you speed, but variable (better) frequency resolution.
    mirr : bool
        method='vacf' only: mirror one-sided VACF at t=0 before fft

    Returns
    -------
    if full_out = False
        | ``(faxis, pdos)``
        | faxis : 1d array [1/unit(dt)]
        | pdos : 1d array, the phonon DOS, normalized to `area`
    if full_out = True
        | if method == 'direct':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx))``
        | if method == 'vavcf':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))``
        |     fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        |     vacf : 1d array, the VACF

    Examples
    --------
from pwtools.constants import fs,rcm_to_Hz
tr = Trajectory(...)
# freq in [Hz] if timestep in [s]
freq,dos = pdos(tr.velocity, m=tr.mass, dt=tr.timestep*fs,
                method='direct', npad=1)
# frequency in [1/cm]
plot(freq/rcm_to_Hz, dos)

    Notes
    -----
    padding (only method='direct'): With `npad` we pad the velocities `vel`
    with ``npad*(nstep-1)`` zeros along `axis` (the time axis) before FFT
    b/c the signal is not periodic. For `npad=1`, this gives us the exact
    same spectrum and frequency resolution as with ``pdos(...,
    method='vacf',mirr=True)`` b/c the array to be fft'ed has length
    ``2*nstep-1`` along the time axis in both cases (remember that the
    array length = length of the time axis influences the freq.
    resolution). FFT is only fast for arrays with length = a power of two.
    Therefore, you may get very different fft speeds depending on whether
    ``2*nstep-1`` is a power of two or not (in most cases it won't). Try
    using `tonext` but remember that you get another (better) frequency
    resolution.

    References
    ----------
    [1] Phys Rev B 47(9) 4863, 1993

    See Also
    --------
    :func:`pwtools.signal.fftsample`
    :func:`pwtools.signal.acorr`
    :func:`direct_pdos`
    :func:`vacf_pdos`

    """
    mass = m
    # assume vel.shape = (nstep,natoms,3)
    axis = 0
    assert vel.shape[-1] == 3
    if mass is not None:
        assert len(mass) == vel.shape[1], "len(mass) != vel.shape[1]"
        # define here b/c may be used twice below
        mass_bc = mass[None,:,None]
    if window:
        sl = [None]*vel.ndim
        sl[axis] = slice(None)  # ':'
        vel2 = vel*(welch(vel.shape[axis])[tuple(sl)])
    else:
        vel2 = vel
    # handle options which are mutually exclusive
    if method == 'vacf':
        assert npad in [0,None], "use npad={0,None} for method='vacf'"
    # padding
    if npad is not None:
        nadd = (vel2.shape[axis]-1)*npad
        if tonext:
            vel2 = pad_zeros(vel2, tonext=True,
                             tonext_min=vel2.shape[axis] + nadd,
                             axis=axis)
        else:
            vel2 = pad_zeros(vel2, tonext=False, nadd=nadd, axis=axis)
    if method == 'direct':
        full_fft_vel = np.abs(fft(vel2, axis=axis))**2.0
        full_faxis = np.fft.fftfreq(vel2.shape[axis], dt)
        split_idx = len(full_faxis)//2
        faxis = full_faxis[:split_idx]
        # First split the array, then multiply by `mass` and average. If
        # full_out, then we need full_fft_vel below, so copy before slicing.
        arr = full_fft_vel.copy() if full_out else full_fft_vel
        fft_vel = num.slicetake(arr, slice(0, split_idx), axis=axis, copy=False)
        if mass is not None:
            fft_vel *= mass_bc
        # average remaining axes, summing is enough b/c normalization is done below
        # sums: (nstep, natoms, 3) -> (nstep, natoms) -> (nstep,)
        pdos = num.sum(fft_vel, axis=axis, keepdims=True)
        default_out = (faxis, num.norm_int(pdos, faxis, area=area))
        if full_out:
            # have to re-calculate this here b/c we never calculate the full_pdos
            # normally
            if mass is not None:
                full_fft_vel *= mass_bc
            full_pdos = num.sum(full_fft_vel, axis=axis, keepdims=True)
            extra_out = (full_faxis, full_pdos, split_idx)
            return default_out + extra_out
        else:
            return default_out
    elif method == 'vacf':
        vacf = fvacf(vel2, m=mass)
        if mirr:
            fft_vacf = fft(mirror(vacf))
        else:
            fft_vacf = fft(vacf)
        full_faxis = np.fft.fftfreq(fft_vacf.shape[axis], dt)
        full_pdos = np.abs(fft_vacf)
        split_idx = len(full_faxis)//2
        faxis = full_faxis[:split_idx]
        pdos = full_pdos[:split_idx]
        default_out = (faxis, num.norm_int(pdos, faxis, area=area))
        extra_out = (full_faxis, full_pdos, split_idx, vacf, fft_vacf)
        if full_out:
            return default_out + extra_out
        else:
            return default_out



#fname = './sim_nve.vel_0.xyz'
#fname = '/Users/calvin/Dropbox/code/MPI/work/results/md/mbe_opt/sim_nve.vel_0.xyz'
#fname = sys.argv[1]
from pwtools.constants import fs, rcm_to_Hz
def get_dos(fname, time_step, method='direct'):
    vel, M = read_data(fname)
    # freq in [Hz] if timestep in [s]
    if method == 'direct':
        freq, dos = pdos(vel, m=M, dt=time_step*fs, tonext=True, npad=2,
                        method='direct')
    else:
        freq, dos = pdos(vel, m=M, dt=time_step*fs, method='vacf', npad=None, mirr=True)
    return freq, dos
# frequency in [1/cm]
'''plt.plot(freq/rcm_to_Hz, dos, linewidth=1)
plt.xlim(0, 4500)
plt.ylim(0, max(dos[30:]))
plt.xticks(np.arange(4501, step=500))
plt.show()'''

try: 
    fname = sys.argv[1]
    mol = 'eigen'
    freq, dos = get_dos(fname, .5, method='vacf')
    idx0, idx1 = 0, 4500
    #idx0, idx1 = 2100, 3100
    #idx0, idx1 = 1700, 2000
    freq = freq/rcm_to_Hz
    index = np.where((freq>=idx0) & (freq<=idx1),True,False)
    #plt.figure()
    fig, ax1 = plt.subplots()
    plt.plot(freq[index], dos[index], linewidth=1, color='b')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    #plt.xlim(0, 3500)
    plt.xlim(idx0, idx1)
    #plt.xticks(np.arange(2, 4.6, step=0.5))
    plt.ylim(0, max(dos[index]))
    ax1.set_yticks([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.xlabel("Frequency [cm$^{-1}$]",fontsize=11)
    plt.show()
    #fig.subplots_adjust(hspace=0)
    #plt.legend()
    # Remove horizontal space between axes

    
except IndexError:
    dir_list = os.listdir(os.getcwd())
    cal_por = False
    if cal_por:
        #fname = 'por/nve/sim_nve.vel_0.xyz'
        fname = 'por/nothermo/sim_nve.vel_0.xyz'
        freq, dos = get_dos(fname, .5, method='vacf')
        idx0, idx1 = 0, 3500
        #idx0, idx1 = 2100, 3100
        #idx0, idx1 = 1700, 2000
        freq = freq/rcm_to_Hz
        index = np.where((freq>=idx0) & (freq<=idx1),True,False)
        plt.figure()
        fig, ax1 = plt.subplots()
        plt.plot(freq[index], dos[index], linewidth=1, color='b')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(500))
        ax1.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        #plt.xlim(0, 3500)
        plt.xlim(idx0, idx1)
        #plt.xticks(np.arange(2, 4.6, step=0.5))
        plt.ylim(0, max(dos[index]))
        ax1.set_yticks([])
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        #fig.subplots_adjust(hspace=0)
        #plt.legend()
        # Remove horizontal space between axes

        plt.xlabel("Frequency [cm$^{-1}$]",fontsize=11)
        #plt.show()
        # Remove horizontal space between axes

        #An inset
        ax2 = plt.axes([0,0,0.5,0.5])
        ip = InsetPosition(ax1, [0.55,0.2,0.35,0.35])
        ax2.set_axes_locator(ip)
        idx0, idx1 = 2100, 3100
        #idx0, idx1 = 1700, 2000
        index = np.where((freq>=idx0) & (freq<=idx1),True,False)
        #ax = plt.subplot(111)
        ax2.plot(freq[index], dos[index], linewidth=1, color='b')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        #ax2.set_xticks(fontsize=7)
        ax2.tick_params(axis='x', which='major', labelsize=7)
        ax2.set_xlim(idx0, idx1)
        ax2.set_ylim(0, max(dos[index]))
        ax2.set_yticks([])
        for sp in ['right', 'top', 'left']:
            ax2.spines[sp].set_linestyle("--")
            ax2.spines[sp].set_color("0.5")
        mark_inset(ax1, ax2, loc1=3, loc2=4, fc="none", ec='0.5', linestyle='--')
        #plt.show()
        plt.savefig('/Users/calvin/Dropbox/code/MPI/work/results/figs/vdos_por.eps')
    else:
        data_dic = {'Normal OSV-MP2':0,
                    'MBE(3)-OSV-MP2':0,
                    'c-MBE(3)-OSV-MP2':0,
                    'g-MBE(3)-OSV-MP2':0}
        for mol in ['eigen', 'zundel']:
            data_list = []
            max_dos = []
            for dir_i in dir_list:
                if 'mbe' in dir_i:
                    fname = '%s/%s_1e-2_0.2/sim_nve.vel_0.xyz'%(dir_i, mol)
                    freq, dos = get_dos(fname, 0.5, method='vacf')
                    max_dos.append(max(dos[30:]))
                    #data_list.append([lable_dic[dir_i], freq, dos])
                    if 'conv' in dir_i:
                        data_dic['Normal OSV-MP2'] = [freq, dos]
                    elif 'no' in dir_i:
                        data_dic['MBE(3)-OSV-MP2'] = [freq, dos]
                    elif 'csg' in dir_i:
                        data_dic['c-MBE(3)-OSV-MP2'] = [freq, dos]
                    else:
                        data_dic['g-MBE(3)-OSV-MP2'] = [freq, dos]
            color_list = ['b', 'g', 'r', 'c']#'tab:orange']
            fig, axs = plt.subplots(len(data_dic), 1, sharex=True)
            #for idx, (mode, freq, dos) in enumerate(data_list):
            idx = 0
            for mode in data_dic.keys():
                freq, dos = data_dic[mode]
                freq = freq/rcm_to_Hz
                idx_list = freq > 0
                axs[idx].plot(freq[idx_list], dos[idx_list], linewidth=1, color=color_list[idx], label=mode)
                axs[idx].set_ylim(0, max(max_dos))
                axs[idx].set_yticks([])
                axs[idx].set_xlim(0, 4500)
                axs[idx].set_xticks(np.arange(4501, step=500))
                axs[idx].spines['right'].set_visible(False)
                axs[idx].spines['top'].set_visible(False)
                axs[idx].spines['left'].set_visible(False)
                axs[idx].legend(loc='upper left',fontsize=9)
                idx += 1
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0)
            plt.xlabel("Frequency [cm$^{-1}$]",fontsize=11)
            #plt.show()
            plt.savefig('/Users/calvin/Dropbox/code/MPI/work/results/figs/vdos_%s.eps'%mol)
