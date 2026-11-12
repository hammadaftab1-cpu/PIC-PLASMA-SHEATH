"""
Demo 1D1V sheath simulation code

Phase A: quasineutral plasma with periodic boundaries

Developed for the Fundamentals of the Particle In Cell Method course
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# constants
EPS_0 = 8.85418782e-12   # F/m, vacuum permittivity
K = 1.38065e-23          # J/K, Boltzmann constant
ME = 9.10938215e-31      # kg, electron mass
QE = 1.602176565e-19     # C, electron charge
AMU = 1.660538921e-27    # kg, atomic mass unit

# simulation parameters
PLASMA_DEN = 1e14        # plasma density to load
NUM_IONS = 100000        # number of ions
NUM_ELECTRONS = 200000   # number of electrons
DX = 1e-4                # cell spacing
NC = 199                 # number of cells
NUM_TS = 1000            # number of time steps
DT = 1e-12               # time step size


# Data structure to hold domain information
class Domain:
    def __init__(self):
        self.ni = 0
        self.x0 = 0.0
        self.dx = 0.0
        self.xl = 0.0
        self.xmax = 0.0

        self.phi = None
        self.ef = None
        self.rho = None
        self.ndi = None
        self.nde = None


# Data structure for particle storage
class Particle:
    def __init__(self, x, v):
        self.x = x
        self.v = v


# Data structure to hold species information
class Species:
    def __init__(self):
        self.mass = 0.0
        self.charge = 0.0
        self.spwt = 0.0

        self.np = 0
        self.np_alloc = 0
        self.part = []


# GLOBAL VARIABLES
domain = Domain()


# scatter particles of species to the mesh
def ScatterSpecies(species, den):
    # initialize densities to zero
    den[:] = 0.0

    # scatter particles to the mesh
    for p in range(species.np):
        lc = XtoL(species.part[p].x)
        scatter(lc, species.spwt, den)

    # apply periodic boundaries
    den[0] += den[domain.ni - 1]
    den[domain.ni - 1] = den[0]

    # divide by cell volume
    for i in range(domain.ni):
        den[i] /= domain.dx


# adds new particle to the species
def AddParticle(species, x, v):
    # abort if we ran out of space to store this particle
    if species.np > species.np_alloc - 1:
        raise SystemExit("Too many particles!")

    # store position and velocity of this particle
    species.part.append(Particle(x, v))

    # increment particle counter
    species.np += 1


# computes charge density by adding ion and electron data
def ComputeRho(ions, electrons):
    rho = domain.rho

    for i in range(domain.ni):
        rho[i] = ions.charge * domain.ndi[i] + electrons.charge * domain.nde[i]

    # remove numerical noise by setting zero when densities less than 1e8/m^3
    if False:
        for i in range(domain.ni):
            if abs(rho[i]) < 1e8 * QE:
                rho[i] = 0.0


# solves potential using the Gauss Seidel Method, returns true if converged
# periodic boundaries version
def SolvePotential(phi, rho):
    dx2 = domain.dx * domain.dx

    # initialize to zero, this is only needed because of periodic b.c. without a dirichlet
    phi[:] = 0.0

    # solve potential
    L2 = 0.0
    for solver_it in range(5000):
        # Gauss Seidel method, phi[i-1]-2*phi[i]+phi[i+1] = -dx^2*rho[i]/eps_0
        for i in range(domain.ni):
            im = i - 1
            if im < 0:
                im = domain.ni - 2

            ip = i + 1
            if ip > domain.ni - 1:
                ip = 1

            phi[i] = 0.5 * (phi[im] + phi[ip] + dx2 * rho[i] / EPS_0)

        # check for convergence
        if solver_it % 25 == 0:
            total = 0.0
            for i in range(domain.ni):
                im = i - 1
                if im < 0:
                    im = domain.ni - 2

                ip = i + 1
                if ip > domain.ni - 1:
                    ip = 1

                R = -rho[i] / EPS_0 - (phi[im] - 2.0 * phi[i] + phi[ip]) / dx2
                total += R * R

            L2 = np.sqrt(total) / domain.ni
            if L2 < 1e-6:
                return True

    print("Gauss-Seidel solver failed to converge, L2=", L2)
    return False


# computes electric field by differentiating potential
def ComputeEF(phi, ef):
    for i in range(1, domain.ni - 1):
        ef[i] = -(phi[i + 1] - phi[i - 1]) / (2.0 * domain.dx)

    # periodic boundaries
    ef[0] = -(phi[1] - phi[domain.ni - 2]) / (2.0 * domain.dx)
    ef[domain.ni - 1] = ef[0]


# moves particles of a single species
def PushSpecies(species, ef):
    # precompute q/m
    qm = species.charge / species.mass

    # loop over particles
    for p in range(species.np):
        # grab this particle
        part = species.part[p]

        # compute particle node position
        lc = XtoL(part.x)

        # gather electric field onto particle position
        part_ef = gather(lc, ef)

        # advance velocity
        part.v += DT * qm * part_ef

        # advance position
        part.x += DT * part.v

        # apply periodic boundaries
        if part.x < domain.x0:
            part.x += domain.xl
        if part.x > domain.xmax:
            part.x -= domain.xl


# rewinds particle velocities by -0.5DT
def RewindSpecies(species, ef):
    # precompute q/m
    qm = species.charge / species.mass

    for p in range(species.np):
        part = species.part[p]

        # compute particle node position
        lc = XtoL(part.x)

        # gather electric field onto particle position
        part_ef = gather(lc, ef)

        # advance velocity
        part.v -= 0.5 * DT * qm * part_ef


# converts physical coordinate to logical
def XtoL(pos):
    li = (pos - domain.x0) / domain.dx
    return li


# scatters scalar value onto a field at logical coordinate lc
def scatter(lc, value, field):
    # Apply periodic boundaries to the logical coordinate.
    # The last mesh node is the duplicate periodic endpoint.
    lc = lc % (domain.ni - 1)
    i = int(lc)
    di = lc - i

    field[i] += value * (1.0 - di)
    field[i + 1] += value * di


# gathers field value at logical coordinate lc
def gather(lc, field):
    # Apply periodic boundaries to the logical coordinate.
    lc = lc % (domain.ni - 1)
    i = int(lc)
    di = lc - i

    # gather field value onto particle position
    val = field[i] * (1.0 - di) + field[i + 1] * di
    return val


# writes new zone to the results file
def WriteResults(ts):
    f_name = "results_" + str(ts) + ".csv"

    with open(f_name, "w") as out:
        out.write("x,nde,ndi,rho,phi,ef\n")

        for i in range(domain.ni):
            out.write(
                str(i * domain.dx) + ","
                + str(domain.nde[i]) + ","
                + str(domain.ndi[i]) + ","
                + str(domain.rho[i]) + ","
                + str(domain.phi[i]) + ","
                + str(domain.ef[i]) + "\n"
            )


# computes species kinetic energy in electron volt
def ComputeKE(species):
    ke = 0.0

    for p in range(species.np):
        ke += species.part[p].v * species.part[p].v

    # multiply by 0.5*mass
    ke *= 0.5 * (species.spwt * species.mass)

    # convert to electron volts, 1eV=QE joules
    ke /= QE

    return ke


# --------- main -------------
def main():
    global domain

    t_start = time.perf_counter()

    # 1) initialize domain
    domain.ni = NC + 1
    domain.dx = DX
    domain.x0 = 0.0
    domain.xl = (domain.ni - 1) * domain.dx
    domain.xmax = domain.x0 + domain.xl

    # allocate data structures
    domain.phi = np.zeros(domain.ni)
    domain.rho = np.zeros(domain.ni)
    domain.ef = np.zeros(domain.ni)
    domain.nde = np.zeros(domain.ni)
    domain.ndi = np.zeros(domain.ni)

    # save arrays
    phi = domain.phi
    rho = domain.rho
    ef = domain.ef
    nde = domain.nde
    ndi = domain.ndi

    # variables to hold species data ions
    ions = Species()
    electrons = Species()

    # set material data
    ions.mass = 8 * AMU
    ions.charge = QE
    ions.spwt = PLASMA_DEN * domain.xl / NUM_IONS
    ions.np = 0
    ions.np_alloc = NUM_IONS

    electrons.mass = ME
    electrons.charge = -QE
    electrons.spwt = PLASMA_DEN * domain.xl / NUM_ELECTRONS
    electrons.np = 0
    electrons.np_alloc = NUM_ELECTRONS

    # load uniformly spaced ions and electrons (quiet start) ( after this data we need to populate these array in the simulatio())
    delta_ions = domain.xl / NUM_IONS
    for p in range(NUM_IONS):
        x = domain.x0 + p * delta_ions
        v = 0.0
        AddParticle(ions, x, v)

    # now do the same for electrons
    delta_electrons = domain.xl / NUM_ELECTRONS
    for p in range(NUM_ELECTRONS):
        x = domain.x0 + p * delta_electrons
        v = 0.0
        AddParticle(electrons, x, v)
# For the leapfrog methdo we need to rewind the vleovity 1/2 step bavkward, but fiorst we ned to oabteidn the eeltric field and velocity fo that we need to mesusre denuisty first
    # compute number density
    ScatterSpecies(ions, ndi)
    ScatterSpecies(electrons, nde)

    # compute charge density and solve potential
    ComputeRho(ions, electrons)
    SolvePotential(phi, rho)
    ComputeEF(phi, ef)

    RewindSpecies(ions, ef)
    RewindSpecies(electrons, ef)

    # OUTPUT
    WriteResults(0)

    # MAIN LOOP
    for ts in range(1, NUM_TS + 1):
        # compute number density
        ScatterSpecies(ions, ndi)
        ScatterSpecies(electrons, nde)

        ComputeRho(ions, electrons)
        SolvePotential(phi, rho)
        ComputeEF(phi, ef)

        # move particles
        PushSpecies(electrons, ef)
        PushSpecies(ions, ef)

        # write diagnostics
        if ts % 50 == 0:
            # max phi
            max_phi = phi[0]
            for i in range(domain.ni):
                if phi[i] > max_phi:
                    max_phi = phi[i]

            # compute kinetic energy
            ke_ions = ComputeKE(ions)
            ke_electrons = ComputeKE(electrons)

            print(
                "TS:", ts,
                "ions: (np=", ions.np, ", KE=", ke_ions,
                ") electrons: (np=", electrons.np, ", KE=", ke_electrons,
                ") delta phi:", max_phi - phi[0]
            )

        # save animation data
        if ts % 500 == 0:
            WriteResults(ts)

    # free up memory
    del phi, rho, ef, nde, ndi

    # free particles
    del ions.part
    del electrons.part

    t_end = time.perf_counter()
    duration = (t_end - t_start) * 1000.0
    print("Simulation took", duration, "ms")


if __name__ == "__main__":
    main()

    # Plot results from the saved output files
    time_steps = [0, 500, 1000]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ts in time_steps:
        data = np.genfromtxt(
            "results_" + str(ts) + ".csv",
            delimiter=",",
            names=True
        )

        axes[0, 0].plot(data["x"], data["nde"], label="Electrons, TS=" + str(ts))
        axes[0, 0].plot(data["x"], data["ndi"], "--", label="Ions, TS=" + str(ts))
        axes[0, 1].plot(data["x"], data["rho"], label="TS=" + str(ts))
        axes[1, 0].plot(data["x"], data["phi"], label="TS=" + str(ts))
        axes[1, 1].plot(data["x"], data["ef"], label="TS=" + str(ts))

    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("Number density (m$^{-3}$)")
    axes[0, 0].set_title("Electron and ion density")
    axes[0, 0].legend()

    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("Charge density (C/m$^3$)")
    axes[0, 1].set_title("Charge density")
    axes[0, 1].legend()

    axes[1, 0].set_xlabel("x (m)")
    axes[1, 0].set_ylabel("Potential (V)")
    axes[1, 0].set_title("Electric potential")
    axes[1, 0].legend()

    axes[1, 1].set_xlabel("x (m)")
    axes[1, 1].set_ylabel("Electric field (V/m)")
    axes[1, 1].set_title("Electric field")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
