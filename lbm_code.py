import os
import sys
import time

import numpy as np
from numba import njit, prange, set_num_threads

import matplotlib.pyplot as plt
import matplotlib.animation as animation

NUM_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 96
set_num_threads(NUM_THREADS)
os.environ["NUMBA_NUM_THREADS"] = str(NUM_THREADS)

CROP = 500
NX, NY, NZ = CROP + 2, CROP + 2, CROP + 2

OMEGA       = 1.07
DENSITY     = 1.0
DELTA_RHO   = 0.0001 / 2
RESOLUTION = 0.00000225
MIN_STEPS   = 3000
CONV_TOL    = 1e-2
PRINT_EVERY = 100

def read_geometry(filename: str, nx=NX, ny=NY, nz=NZ):
    print("Reading raw geometry …", flush=True)
    t0 = time.time()

    RAW_N = 1000
    CROP_N = nx - 2

    dtype = np.uint8

    raw = np.memmap(filename, dtype=dtype, mode="r", shape=(RAW_N, RAW_N, RAW_N))

    s = (RAW_N - CROP_N) // 2
    e = s + CROP_N

    crop = np.array(raw[s:e, s:e, s:e], dtype=np.int32)

    print("Crop shape:", crop.shape)
    print("Unique values:", np.unique(crop))

    geom = np.zeros(nx * ny * nz, dtype=np.int32)
    geom3 = geom.reshape(nz, ny, nx)

    geom3[1:-1, 1:-1, 1:-1] = crop

    geom3[:, :, 0] = 1
    geom3[:, :, -1] = 1
    geom3[:, 0, :] = 1
    geom3[:, -1, :] = 1

    interior = geom3[1:-1, 1:-1, 1:-1]
    porosity = float(np.sum(interior == 0)) / interior.size

    print(f"Loaded in {time.time()-t0:.1f}s | porosity = {porosity*100:.3f}%")
    return geom, porosity

@njit(parallel=True, cache=True, fastmath=True)
def collision(F, geom, ux, uy, uz, ro,
              nx, ny, nz, omega, density, delta_rho):
    T1  = 1.0 / 3.0
    T2  = 1.0 / 18.0
    T3  = 1.0 / 36.0
    om1 = 1.0 - omega

    for p in prange(nx * ny * nz):
        f0  = F[p,  0];  f1  = F[p,  1];  f2  = F[p,  2];  f3  = F[p,  3]
        f4  = F[p,  4];  f5  = F[p,  5];  f6  = F[p,  6];  f7  = F[p,  7]
        f8  = F[p,  8];  f9  = F[p,  9];  f10 = F[p, 10];  f11 = F[p, 11]
        f12 = F[p, 12];  f13 = F[p, 13];  f14 = F[p, 14];  f15 = F[p, 15]
        f16 = F[p, 16];  f17 = F[p, 17];  f18 = F[p, 18]

        rho = (f0  + f1  + f2  + f3  + f4  + f5  + f6  + f7  + f8  + f9  +
               f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18)
        ro[p] = rho

        if geom[p] == 1:
            ux[p] = 0.0;  uy[p] = 0.0;  uz[p] = 0.0
            F[p,  0] = omega * T1 * rho + om1 * f0
            ff = omega * T2 * rho
            fe = omega * T3 * rho
            F[p,  1] = ff + om1*f1;   F[p, 10] = ff + om1*f10
            F[p,  2] = ff + om1*f2;   F[p, 11] = ff + om1*f11
            F[p,  3] = ff + om1*f3;   F[p, 12] = ff + om1*f12
            F[p,  4] = fe + om1*f4;   F[p, 13] = fe + om1*f13
            F[p,  5] = fe + om1*f5;   F[p, 14] = fe + om1*f14
            F[p,  6] = fe + om1*f6;   F[p, 15] = fe + om1*f15
            F[p,  7] = fe + om1*f7;   F[p, 16] = fe + om1*f16
            F[p,  8] = fe + om1*f8;   F[p, 17] = fe + om1*f17
            F[p,  9] = fe + om1*f9;   F[p, 18] = fe + om1*f18
            continue

        inv_rho = 1.0 / rho
        vx = (f10 + f13 + f14 + f15 + f16 - f1  - f4  - f5  - f6  - f7 ) * inv_rho
        vy = (f11 + f13 + f5  + f17 + f18 - f2  - f4  - f14 - f8  - f9 ) * inv_rho
        vz = (f12 + f15 + f7  + f17 + f9  - f3  - f16 - f6  - f18 - f8 ) * inv_rho

        k = p // (nx * ny)
        if k == 0:
            rho = density + delta_rho;  ro[p] = rho
            vx  = 0.0;  vy = 0.0
            vz  = (f12+f15+f7+f17+f9 - f3-f16-f6-f18-f8) / rho
        elif k == nz - 1:
            rho = density - delta_rho;  ro[p] = rho
            vx  = 0.0;  vy = 0.0
            vz  = (f12+f15+f7+f17+f9 - f3-f16-f6-f18-f8) / rho

        ux[p] = vx;  uy[p] = vy;  uz[p] = vz

        u2   = vx*vx + vy*vy + vz*vz
        base = 1.0 - 1.5 * u2

        F[p,  0] = omega*(T1*rho*base) + om1*f0

        e = vx
        F[p, 10] = omega*(T2*rho*(base + 3.0*e + 4.5*e*e)) + om1*f10
        F[p,  1] = omega*(T2*rho*(base - 3.0*e + 4.5*e*e)) + om1*f1
        e = vy
        F[p, 11] = omega*(T2*rho*(base + 3.0*e + 4.5*e*e)) + om1*f11
        F[p,  2] = omega*(T2*rho*(base - 3.0*e + 4.5*e*e)) + om1*f2
        e = vz
        F[p, 12] = omega*(T2*rho*(base + 3.0*e + 4.5*e*e)) + om1*f12
        F[p,  3] = omega*(T2*rho*(base - 3.0*e + 4.5*e*e)) + om1*f3

        e = vx+vy; F[p, 13] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f13
        e = vx-vy; F[p, 14] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f14
        e =-vx+vy; F[p,  5] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f5
        e =-vx-vy; F[p,  4] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f4

        e = vx+vz; F[p, 15] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f15
        e = vx-vz; F[p, 16] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f16
        e =-vx+vz; F[p,  7] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f7
        e =-vx-vz; F[p,  6] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f6

        e = vy+vz; F[p, 17] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f17
        e = vy-vz; F[p, 18] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f18
        e =-vy+vz; F[p,  9] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f9
        e =-vy-vz; F[p,  8] = omega*(T3*rho*(base+3.0*e+4.5*e*e)) + om1*f8

@njit(parallel=True, cache=True, fastmath=True)
def streaming(F, F_new, geom, nx, ny, nz):
    CX  = ( 0,-1, 0, 0,-1,-1,-1,-1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0)
    CY  = ( 0, 0,-1, 0,-1, 1, 0, 0,-1,-1, 0, 1, 0, 1,-1, 0, 0, 1, 1)
    CZ  = ( 0, 0, 0,-1, 0, 0,-1, 1,-1, 1, 0, 0, 1, 0, 0, 1,-1, 1,-1)
    OPP = ( 0,10,11,12,13,14,15,16,17,18, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    for p in prange(nx * ny * nz):
        k  =  p  // (nx * ny)
        jj = (p  %  (nx * ny)) // nx
        ii =  p  %  nx

        for d in range(19):
            si = ii - CX[d]
            sj = jj - CY[d]
            sk =  k - CZ[d]

            if si < 0 or si >= nx or sj < 0 or sj >= ny or sk < 0 or sk >= nz:
                F_new[p, d] = F[p, OPP[d]]
            else:
                src = sk * nx * ny + sj * nx + si
                if geom[src] == 1:
                    F_new[p, d] = F[p, OPP[d]]
                else:
                    F_new[p, d] = F[src, d]

@njit(parallel=True, cache=True, fastmath=True)
def zou_he_inlet(F, geom, nx, ny, density, delta_rho):
    rho_in = density + delta_rho

    for p in prange(nx * ny):
        ii = p % nx
        jj = p // nx
        if ii == 0 or ii == nx-1 or jj == 0 or jj == ny-1:
            continue
        if geom[p] == 1:
            continue

        f0  = F[p,  0];  f1  = F[p,  1];  f2  = F[p,  2];  f3  = F[p,  3]
        f4  = F[p,  4];  f5  = F[p,  5];  f6  = F[p,  6];  f8  = F[p,  8]
        f10 = F[p, 10];  f11 = F[p, 11];  f13 = F[p, 13];  f14 = F[p, 14]
        f16 = F[p, 16];  f18 = F[p, 18]

        known = f0 + f1 + f2 + f10 + f11 + f13 + f4 + f5 + f14
        out   = f3 + f16 + f6 + f18 + f8

        uz_in = 1.0 - (known + 2.0 * out) / rho_in

        dx = (f10 + f13 + f14) - (f1 + f5 + f4)
        dy = (f11 + f13 +  f5) - (f2 + f4 + f14)

        q = rho_in * uz_in
        F[p, 12] = f3  + q / 3.0
        F[p, 15] = f6  + q / 6.0 - 0.5 * dx
        F[p,  7] = f16 + q / 6.0 + 0.5 * dx
        F[p, 17] = f8  + q / 6.0 - 0.5 * dy
        F[p,  9] = f18 + q / 6.0 + 0.5 * dy

@njit(parallel=True, cache=True, fastmath=True)
def zou_he_outlet(F, geom, nx, ny, nz, density, delta_rho):
    rho_out = density - delta_rho
    base_p  = (nz - 1) * nx * ny

    for p_local in prange(nx * ny):
        p  = base_p + p_local
        ii = p_local % nx
        jj = p_local // nx
        if ii == 0 or ii == nx-1 or jj == 0 or jj == ny-1:
            continue
        if geom[p] == 1:
            continue

        f0  = F[p,  0];  f1  = F[p,  1];  f2  = F[p,  2]
        f4  = F[p,  4];  f5  = F[p,  5];  f7  = F[p,  7]
        f9  = F[p,  9];  f10 = F[p, 10];  f11 = F[p, 11]
        f12 = F[p, 12];  f13 = F[p, 13];  f14 = F[p, 14]
        f15 = F[p, 15];  f17 = F[p, 17]

        known = f0 + f1 + f2 + f10 + f11 + f13 + f4 + f5 + f14
        inn   = f12 + f15 + f7 + f17 + f9

        uz_out = -1.0 + (known + 2.0 * inn) / rho_out

        dx = (f10 + f13 + f14) - (f1 + f5 + f4)
        dy = (f11 + f13 +  f5) - (f2 + f4 + f14)

        q = rho_out * uz_out
        F[p,  3] = f12 - q / 3.0
        F[p, 16] = f7  - q / 6.0 - 0.5 * dx
        F[p,  6] = f15 - q / 6.0 + 0.5 * dx
        F[p, 18] = f9  - q / 6.0 - 0.5 * dy
        F[p,  8] = f17 - q / 6.0 + 0.5 * dy

@njit(parallel=True, cache=True, fastmath=True)
def compute_avg_uz(uz, geom, nx, ny, nz):
    total = 0.0
    denom = float((nx - 2) * (ny - 2) * nz)

    for p in prange(nx * ny * nz):
        k  =  p  // (nx * ny)
        jj = (p  %  (nx * ny)) // nx
        ii =  p  %  nx
        if (1 <= k < nz-1 and 1 <= jj < ny-1 and 1 <= ii < nx-1
                and geom[p] == 0):
            total += uz[p]

    return total / denom

def darcy_permeability(avg_uz, porosity, density, omega, resolution, nz, delta_rho):
    nu_factor = 1.0 / omega - 0.5
    K = (1e15 * resolution**2
         * porosity * avg_uz * density * nu_factor * nz
         / delta_rho)
    return K

def save_outputs(ux, uy, uz, ro, perm_log, nx=NX, ny=NY, nz=NZ):
    # np.save("velocity_ux.npy", ux.reshape(nz, ny, nx))
    # np.save("velocity_uy.npy", uy.reshape(nz, ny, nx))
    # np.save("velocity_uz.npy", uz.reshape(nz, ny, nx))
    # np.save("density.npy",     ro.reshape(nz, ny, nx))
    np.savetxt("perm_history.csv", perm_log,
                header="permeability_mD", comments="", fmt="%.10f")
    # print("Saved: velocity_u{x,y,z}.npy  density.npy  perm_history.csv")

ANIM_EVERY = 30
ANIM_K = NZ // 2
ANIM_FILE = "velocity_cross_section.gif"
ANIM_FRAMES = []

def capture_velocity_frame(ux, uy, uz, geom, step, nx=NX, ny=NY, nz=NZ):
    ux3 = ux.reshape(nz, ny, nx)
    uy3 = uy.reshape(nz, ny, nx)
    uz3 = uz.reshape(nz, ny, nx)
    g3 = geom.reshape(nz, ny, nx)

    speed = np.sqrt(ux3**2 + uy3**2 + uz3**2)

    k = nz // 2
    j = ny // 2
    i = nx // 2

    xy = speed[k, :, :].copy()
    xz = speed[:, j, :].copy()
    yz = speed[:, :, i].copy()

    xy[g3[k, :, :] == 1] = np.nan
    xz[g3[:, j, :] == 1] = np.nan
    yz[g3[:, :, i] == 1] = np.nan

    return step, xy, xz, yz

def save_velocity_animation(frames, filename="velocity_mid_sections.gif"):
    if len(frames) == 0:
        print("No animation frames saved.")
        return

    vmax = 0.0
    for _, xy, xz, yz in frames:
        vmax = max(vmax, np.nanmax(xy), np.nanmax(xz), np.nanmax(yz))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(frames[0][1], origin="lower", vmin=0, vmax=vmax)
    im1 = axes[1].imshow(frames[0][2], origin="lower", vmin=0, vmax=vmax)
    im2 = axes[2].imshow(frames[0][3], origin="lower", vmin=0, vmax=vmax)

    axes[0].set_title("XY mid-section")
    axes[1].set_title("XZ mid-section")
    axes[2].set_title("YZ mid-section")

    for ax in axes:
        ax.set_xlabel("index")
        ax.set_ylabel("index")

    cb = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.85)
    cb.set_label("Velocity magnitude")

    title = fig.suptitle(f"Velocity magnitude | step {frames[0][0]}")

    def update(n):
        step, xy, xz, yz = frames[n]
        im0.set_data(xy)
        im1.set_data(xz)
        im2.set_data(yz)
        title.set_text(f"Velocity magnitude | 3 mid-sections | step {step}")
        return im0, im1, im2, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=150,
        blit=False
    )

    ani.save(filename, writer="pillow", fps=10, dpi=150)
    plt.close(fig)
    print(f"Saved animation: {filename}")

MAX_STEPS = 1_000

def main():
    ncells = NX * NY * NZ
    mem_gb = 2 * ncells * 19 * 8 / 1e9

    banner = (f"D3Q19 LBM  |  grid {NX}×{NY}×{NZ}  |  "
              f"{NUM_THREADS} threads  |  ~{mem_gb:.1f} GB RAM")
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    geom, porosity = read_geometry("/lustre01/other/2685my/DGRR/DRP-317/Berea/Berea_2d25um_binary.raw/Berea_2d25um_binary.raw")

    print(f"\nAllocating {mem_gb:.1f} GB …", flush=True)
    F     = np.full((ncells, 19), 1.0 / 19.0, dtype=np.float64)
    F_new = np.empty_like(F)
    ux    = np.zeros(ncells, dtype=np.float64)
    uy    = np.zeros(ncells, dtype=np.float64)
    uz    = np.zeros(ncells, dtype=np.float64)
    ro    = np.ones (ncells, dtype=np.float64)

    print("Compiling Numba kernels (first run only) …", flush=True)
    t_compile = time.time()
    collision (F, geom, ux, uy, uz, ro, NX, NY, NZ, OMEGA, DENSITY, DELTA_RHO)
    streaming (F, F_new, geom, NX, NY, NZ);  F, F_new = F_new, F
    zou_he_inlet (F, geom, NX, NY, DENSITY, DELTA_RHO)
    zou_he_outlet(F, geom, NX, NY, NZ, DENSITY, DELTA_RHO)
    compute_avg_uz(uz, geom, NX, NY, NZ)
    print(f"   Compiled in {time.time()-t_compile:.1f}s", flush=True)

    F[:]  = 1.0 / 19.0
    ux[:] = 0.0;  uy[:] = 0.0;  uz[:] = 0.0;  ro[:] = 1.0

    permeability = 0.0
    perm_prev    = 1e10
    perm_log     = []
    step         = 0
    t_start      = time.time()

    print(f"\n{'Step':>7}  {'Perm [mD]':>12}  {'|ΔPerm|':>11}  {'Elapsed':>9}")
    print("-" * 48)

    while ((abs(perm_prev - permeability) > CONV_TOL or step < MIN_STEPS) and step < MAX_STEPS):
        perm_prev = permeability
        step += 1

        collision(F, geom, ux, uy, uz, ro,
                  NX, NY, NZ, OMEGA, DENSITY, DELTA_RHO)

        streaming(F, F_new, geom, NX, NY, NZ)
        F, F_new = F_new, F

        zou_he_inlet (F, geom, NX, NY, DENSITY, DELTA_RHO)
        zou_he_outlet(F, geom, NX, NY, NZ, DENSITY, DELTA_RHO)

        avg_uz_val   = compute_avg_uz(uz, geom, NX, NY, NZ)
        permeability = darcy_permeability(
            avg_uz_val, porosity, DENSITY, OMEGA, RESOLUTION, NZ, DELTA_RHO)
        perm_log.append(permeability)

        if step % PRINT_EVERY == 0 or step == 1:
            elapsed = time.time() - t_start
            delta   = abs(permeability - perm_prev)
            print(f"{step:>7}  {permeability:>12.4f}  {delta:>11.6f}  {elapsed:>8.1f}s",
                  flush=True)
        
        if step % ANIM_EVERY == 0 or step == 1:
            ANIM_FRAMES.append(capture_velocity_frame(
                ux, uy, uz, geom, step, NX, NY, NZ
            ))

    elapsed = time.time() - t_start
    print("\n" + "=" * 48)
    print(f"Converged after {step} steps  ({elapsed:.1f} s total)")
    print(f"Porosity     : {porosity * 100:.3f} %")
    print(f"Permeability : {permeability:.4f} mD")
    steps_per_sec = step / elapsed
    print(f"Throughput   : {steps_per_sec:.2f} steps/s  "
          f"({ncells * steps_per_sec / 1e6:.1f} Mcell·updates/s)")
    print("=" * 48)

    save_velocity_animation(ANIM_FRAMES)
    save_outputs(ux, uy, uz, ro, perm_log)

if __name__ == "__main__":
    main()