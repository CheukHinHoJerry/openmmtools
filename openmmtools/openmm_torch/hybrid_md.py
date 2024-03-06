import sys
from ase.io import read, write
import random
import mdtraj
import torch
import time
import openmm.app as app


from mace.calculators import MACECalculator
import numpy as np
from tempfile import mkstemp
from ase import Atoms
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolFromXYZFile
from openmm.openmm import System
from typing import List, Tuple, Optional
from openmm.app.internal.unitcell import reducePeriodicBoxVectors
from openmm import (
    LangevinMiddleIntegrator,
    RPMDIntegrator,
    MonteCarloBarostat,
    CustomTorsionForce,
    NoseHooverIntegrator,
    VerletIntegrator,
    RPMDMonteCarloBarostat,
    CMMotionRemover,
)
import matplotlib.pyplot as plt
from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
from openmmtools import states, mcmc
from openmmtools.multistate.replicaexchange import ReplicaExchangeSampler
from mdtraj.reporters import HDF5Reporter, NetCDFReporter
from mdtraj.geometry.dihedral import indices_phi, indices_psi
from openmm.app import (
    Simulation,
    StateDataReporter,
    PDBReporter,
    DCDReporter,
    CheckpointReporter,
    PDBFile,
    Modeller,
    CutoffNonPeriodic,
    PME,
    HBonds,
)
from ase.optimize import LBFGS
from openmm.app.metadynamics import Metadynamics, BiasVariable
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.unit import (
    kelvin,
    picosecond,
    kilocalorie_per_mole,
    femtosecond,
    kilojoule_per_mole,
    picoseconds,
    femtoseconds,
    bar,
    nanometers,
    molar,
    angstrom,
    MOLAR_GAS_CONSTANT_R,    
)
from openff.toolkit.topology import Molecule
from openff.toolkit import ForceField

from openmmtools import alchemy

from openmmml.models.macepotential import MACEPotentialImplFactory
from openmmml.models.anipotential import ANIPotentialImplFactory
from openmmml import MLPotential

from openmmtools.openmm_torch.repex import (
    MixedSystemConstructor,
    RepexConstructor,
    get_atoms_from_resname,
)
from openmmtools.openmm_torch.utils import (
    initialize_mm_forcefield,
    set_smff,
)
from tempfile import mkstemp
import os
import logging
from abc import ABC, abstractmethod

# === my imports
from .sd_integrators import CustomLangevinIntegrator
from .integrator_mt import CustomLangevinIntegrator_mt
from .parareal_utils import *
import threading, copy


def get_xyz_from_mol(mol):
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return xyz


# def parareal_error_novel(N_init, N_final, x_prev, x_cur, v_prev, v_cur):
#     return sum([np.linalg.norm(x_prev[n] - x_cur[n]) for n in range(N_init, N_final)]) \
#         / sum([np.linalg.norm(x_prev[n]) for n in range(N_init, N_final)])
        
# def parareal_error_withvel(N_init, N_final, x_prev, x_cur, v_prev, v_cur):
#     return sum([np.sqrt(np.linalg.norm(x_prev[n] - x_cur[n]) ** 2 + \
#         np.linalg.norm(v_prev[n] - v_cur[n]) ** 2) for n in range(N_init, N_final)]) \
#         / sum([np.sqrt(np.linalg.norm(x_prev[n]) ** 2 + np.linalg.norm(v_prev[n]) ** 2) for n in range(N_init, N_final)])        

def parareal_error(N_init, N_final, x_prev, x_cur):
    return sum([np.linalg.norm(x_prev[n] - x_cur[n]) for n in range(N_init, N_final)]) \
        / sum([np.linalg.norm(x_prev[n]) for n in range(N_init, N_final)])

MLPotential.registerImplFactory("mace", MACEPotentialImplFactory())
MLPotential.registerImplFactory("ani2x", ANIPotentialImplFactory())

logger = logging.getLogger("INFO")


class MACESystemBase(ABC):
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    system: System
    mm_only: bool
    nl: str
    max_n_pairs: int
    remove_cmm: bool
    unwrap: bool
    set_temperature: bool

    def __init__(
        self,
        file: str,
        model_path: str,
        potential: str,
        output_dir: str,
        temperature: float,
        nl: str,
        max_n_pairs: int,
        minimiser: str,
        pressure: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        mm_only: bool = False,
        remove_cmm: bool = False,
        unwrap: bool = False,
        set_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.file = file
        self.model_path = model_path
        self.potential = potential
        self.temperature = temperature
        self.pressure = pressure
        self.friction_coeff = friction_coeff / picosecond
        self.timestep = timestep * femtosecond
        self.dtype = dtype
        self.nl = nl
        self.max_n_pairs = max_n_pairs
        self.set_temperature = set_temperature
        self.output_dir = output_dir
        self.remove_cmm = remove_cmm
        self.mm_only = mm_only
        self.minimiser = minimiser
        self.unwrap = unwrap
        self.openmm_precision = "Double" if dtype == torch.float64 else "Mixed"
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.SM_FF = set_smff(smff)
        logger.info(f"Using SMFF {self.SM_FF}")

        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_ase_atoms(self, ml_mol: str) -> Tuple[Atoms, Molecule]:
        """Generate the ase atoms object from the

        :param str ml_mol: file path or smiles
        :return Tuple[Atoms, Molecule]: ase Atoms object and initialised openFF molecule
        """
        # ml_mol can be a path to a file, or a smiles string
        if os.path.isfile(ml_mol):
            if ml_mol.endswith(".pdb"):
                # openFF refuses to work with pdb or xyz files, rely on rdkit to do the convertion to a mol first
                molecule = MolFromPDBFile(ml_mol)
                logger.warning(
                    "Initializing topology from pdb - this can lead to valence errors, check your starting structure carefully!"
                )
                molecule = Molecule.from_rdkit(
                    molecule, hydrogens_are_explicit=True, allow_undefined_stereo=True
                )
            elif ml_mol.endswith(".xyz"):
                molecule = MolFromXYZFile(ml_mol)
                molecule = Molecule.from_rdkit(molecule, hydrogens_are_explicit=True)
            else:
                # assume openFF will handle the format otherwise
                molecule = Molecule.from_file(ml_mol, allow_undefined_stereo=True)
        else:
            try:
                molecule = Molecule.from_smiles(ml_mol)
            except:
                raise ValueError(
                    f"Attempted to interpret arg {ml_mol} as a SMILES string, but could not parse"
                )

        _, tmpfile = mkstemp(suffix=".xyz")
        molecule._to_xyz_file(tmpfile)
        atoms = read(tmpfile)
        # os.remove(tmpfile)
        return atoms, molecule

    @abstractmethod
    def create_system(self):
        pass

    def run_mixed_md(
        self,
        steps: int,
        interval: int,
        output_file: str,
        restart: bool,
        run_metadynamics: bool = False,
        integrator_name: str = "langevin",
    ):
        """Runs plain MD on the mixed system, writes a pdb trajectory

        :param int steps: number of steps to run the simulation for
        :param int interval: reportInterval attached to reporters
        """
        if integrator_name == "langevin":
            integrator = LangevinMiddleIntegrator(
                self.temperature, self.friction_coeff, self.timestep
            )
        elif integrator_name == "nose-hoover":
            integrator = NoseHooverIntegrator(
                self.temperature, self.friction_coeff, self.timestep
            )
        elif integrator_name == "verlet":
            integrator = VerletIntegrator(self.timestep)
        elif integrator_name == "rpmd":
            # note this requires a few changes to how we set positions
            integrator = RPMDIntegrator(
                8, self.temperature, self.friction_coeff, self.timestep
            )
        else:
            raise ValueError(
                f"Unrecognized integrator name {integrator_name}, must be one of ['langevin', 'nose-hoover', 'rpmd', 'verlet']"
            )
        if self.remove_cmm:
            logger.info("Using CMM remover")
            self.system.addForce(CMMotionRemover())
        # optionally run NPT with and MC barostat
        if self.pressure is not None:
            if integrator_name == "rpmd":
                # add the special RPMD barostat
                logger.info("Using RPMD barostat")
                barostat = RPMDMonteCarloBarostat(self.pressure, 25)
            else:
                barostat = MonteCarloBarostat(self.pressure, self.temperature)
            self.system.addForce(barostat)

        if run_metadynamics:
            # if we have initialized from xyz, the topology won't have the information required to identify the cv indices, create from a pdb
            input_file = PDBFile(self.file)
            topology = input_file.getTopology()
            meta = self.run_metadynamics(
                topology=topology
                # cv1_dsl_string=self.cv1_dsl_string, cv2_dsl_string=self.cv2_dsl_string
            )
        # set alchemical state

        logger.debug(f"Running mixed MD for {steps} steps")
        simulation = Simulation(
            self.modeller.topology,
            self.system,
            integrator,
            platformProperties={"Precision": self.openmm_precision},
        )
        checkpoint_filepath = os.path.join(self.output_dir, output_file[:-4] + ".chk")
        if restart and os.path.isfile(checkpoint_filepath):
            with open(checkpoint_filepath, "rb") as f:
                logger.info("Loading simulation from checkpoint file...")
                simulation.context.loadCheckpoint(f.read())
        else:
            if isinstance(integrator, RPMDIntegrator):
                for copy in range(integrator.getNumCopies()):
                    integrator.setPositions(copy, self.modeller.getPositions())
            else:
                simulation.context.setPositions(self.modeller.getPositions())
                # rpmd requires that the integrator be used to set positions
            if self.minimiser == "openmm":
                logging.info("Minimising energy...")
                simulation.minimizeEnergy(maxIterations=10)
                if isinstance(integrator, RPMDIntegrator):
                    minimised_state = integrator.getState(
                        0, getPositions=True, getVelocities=True, getForces=True
                    )
                else:
                    minimised_state = simulation.context.getState(
                        getPositions=True, getVelocities=True, getForces=True
                    )

                with open(
                    os.path.join(self.output_dir, "minimised_system.pdb"), "w"
                ) as f:
                    PDBFile.writeFile(
                        self.modeller.topology, minimised_state.getPositions(), file=f
                    )
            else:
                logger.info("Skipping minimisation step")

        if self.set_temperature:
            logger.info(f"Setting temperature to {self.temperature} K")
            simulation.context.setVelocitiesToTemperature(self.temperature)
        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            totalEnergy=True,
            potentialEnergy=True,
            density=True,
            volume=True,
            temperature=True,
            speed=True,
            progress=True,
            totalSteps=steps,
            remainingTime=True,
        )
        simulation.reporters.append(reporter)
        # keep periodic box off to make quick visualisation easier
        simulation.reporters.append(
            PDBReporter(
                file=os.path.join(self.output_dir, output_file),
                reportInterval=interval,
                enforcePeriodicBox=False if self.unwrap else True,
            )
        )
        simulation.reporters.append(DCDReporter(
            file=os.path.join(self.output_dir, "output.dcd"),
            reportInterval=interval,
            append=restart,
            enforcePeriodicBox=False if self.unwrap else True,
            )
        )

        # Add an extra hash to any existing checkpoint files
        checkpoint_files = [f for f in os.listdir(self.output_dir) if f.endswith("#")]
        for file in checkpoint_files:
            os.rename(
                os.path.join(self.output_dir, file),
                os.path.join(self.output_dir, f"{file}#"),
            )

        # backup the existing checkpoint file
        if os.path.isfile(checkpoint_filepath):
            os.rename(checkpoint_filepath, checkpoint_filepath + "#")
        checkpoint_reporter = CheckpointReporter(
            file=checkpoint_filepath, reportInterval=interval
        )
        simulation.reporters.append(checkpoint_reporter)

        if run_metadynamics:
            logger.info("Running metadynamics")
            # handles running the simulation with metadynamics
            meta.step(simulation, steps)

            fe = meta.getFreeEnergy()
            fig, ax = plt.subplots(1, 1)
            ax.imshow(fe)
            fig.savefig(os.path.join(self.output_dir, "free_energy.png"))
            # also write the numpy array to disk
            np.save(os.path.join(self.output_dir, "free_energy.npy"), fe)

        else:
            #simulation.step(steps)
            total_dof = getdof(system=self.system)
            tmp_array = []
            for _ in range(0, steps):
                simulation.step(1)
                state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
                _temp = (2*state.getKineticEnergy()/(total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin)
                tmp_array.append(_temp)
            np.savetxt(os.path.join(self.output_dir, "temperature_array.txt"), tmp_array)

            

    def run_repex(
        self,
        replicas: int,
        restart: bool,
        decouple: bool,
        steps: int,
        steps_per_mc_move: int = 1000,
        steps_per_equilibration_interval: int = 1000,
        equilibration_protocol: str = "minimise",
        checkpoint_interval: int = 10,
        repex_type: str = "interpolate",
    ) -> None:
        repex_file_exists = os.path.isfile(os.path.join(self.output_dir, "repex.nc"))
        # even if restart has been set, disable if the checkpoint file was not found, enforce minimising the system
        if not repex_file_exists:
            restart = False
        if repex_type == "interpolate":
            sampler = RepexConstructor(
                mixed_system=self.system,
                initial_positions=self.modeller.getPositions(),
                intervals_per_lambda_window=2 * replicas,
                steps_per_equilibration_interval=steps_per_equilibration_interval,
                equilibration_protocol=equilibration_protocol,
                temperature=self.temperature * kelvin,
                n_states=replicas,
                restart=restart,
                decouple=decouple,
                mcmc_moves_kwargs={
                    "timestep": 1.0 * femtoseconds,
                    "collision_rate": 10.0 / picoseconds,
                    "n_steps": steps_per_mc_move,
                    "reassign_velocities": False,
                    "n_restart_attempts": 20,
                },
                replica_exchange_sampler_kwargs={
                    "number_of_iterations": steps,
                    "online_analysis_interval": 10,
                    "online_analysis_minimum_iterations": 10,
                },
                storage_kwargs={
                    "storage": os.path.join(self.output_dir, "repex.nc"),
                    "checkpoint_interval": checkpoint_interval,
                    "analysis_particle_indices": get_atoms_from_resname(
                        topology=self.modeller.topology,
                        nnpify_id=self.resname,
                        nnpify_type=self.nnpify_type,
                    ),
                },
            ).sampler
        elif repex_type == "temperature":
            protocol = {"temperature": [300, 310, 330, 370, 450] * kelvin}
            thermo_states = states.create_thermodynamic_state_protocol(
                self.system, protocol
            )
            sampler_states = [
                states.SamplerState(positions=self.modeller.getPositions())
                for _ in thermo_states
            ]
            langevin_move = mcmc.LangevinSplittingDynamicsMove(
                timestep=self.timestep * femtoseconds, n_steps=steps
            )

            sampler = ReplicaExchangeSampler(
                thermo_states, sampler_states, langevin_move
            )

        # do not minimsie if we are hot-starting the simulation from a checkpoint
        if not restart and equilibration_protocol == "minimise":
            logging.info("Minimizing system...")
            t1 = time.time()
            sampler.minimize()
            # just run a few steps to make sure the system is in a reasonable conformation

            logging.info(f"Minimised system  in {time.time() - t1} seconds")
            # we want to write out the positions after the minimisation - possibly something weird is going wrong here and it's ending up in a weird conformation

        sampler.run()

    def run_metadynamics(
        # self, topology: Topology, cv1_dsl_string: str, cv2_dsl_string: str
        self,
        topology: Topology,
    ) -> Metadynamics:
        # run well-tempered metadynamics
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

        cv1_atom_indices = indices_psi(mdtraj_topology)[1]
        cv2_atom_indices = indices_phi(mdtraj_topology)[1]
        logger.info(f"Selcted cv1 torsion atoms {cv1_atom_indices}")
        # logger.info(f"Selcted cv2 torsion atoms {cv2_atom_indices}")
        # takes the mixed system parametrised in the init method and performs metadynamics
        # in the canonical case, this should just use the psi-phi backbone angles of the peptide

        cv1 = CustomTorsionForce("theta")
        # cv1.addTorsion(cv1_atom_indices)
        cv1.addTorsion(*cv1_atom_indices)
        phi = BiasVariable(cv1, -np.pi, np.pi, biasWidth=0.5, periodic=True)

        cv2 = CustomTorsionForce("theta")
        cv2.addTorsion(*cv2_atom_indices)
        psi = BiasVariable(cv2, -np.pi, np.pi, biasWidth=0.5, periodic=True)
        os.makedirs(os.path.join(self.output_dir, "metaD"), exist_ok=True)
        meta = Metadynamics(
            self.system,
            [psi, phi],
            temperature=self.temperature,
            biasFactor=100.0,
            height=1.0 * kilojoule_per_mole,
            frequency=100,
            biasDir=os.path.join(self.output_dir, "metaD"),
            saveFrequency=100,
        )

        return meta



    def run_parareal(
        self,
        steps,
        fine_time_step,
        seed = 123 ,
        equilibrate_steps = 5000,
        delta_conv_x = 1e-5,
        delta_conv_v = 2e-3,
        delta_expl = 5,
        NumParaReal = 125,
    ):
        
        print("=== running parareal simulation ===")

        # gather some info from self
        coarse_time_step = self.timestep
        
        # === init variables ===
        NumCoarseSteps = steps
        NumFineSteps = coarse_time_step / fine_time_step
        NumFineSteps = int(NumFineSteps)
        fine_steps_total = int(NumFineSteps * NumCoarseSteps)
        npos = len(self.modeller.getPositions())
        N_init = 0
        N = NumCoarseSteps + 1
        N_final = copy.deepcopy(N)
        
        ##
        
        # === create integrator ===
        fine_integrators = [CustomLangevinIntegrator_mt(self.temperature, self.friction_coeff, timestep = fine_time_step, NumFineSteps = NumFineSteps) for _ in range(NumCoarseSteps)]
        integrator = CustomLangevinIntegrator(self.temperature, self.friction_coeff, coarse_time_step)
        
        logger.debug(f"Running mixed MD for {steps} steps")
        
        coarse_simulation = Simulation(
            self.modeller.topology,
            self.coarse_system,
            integrator,
            platformProperties={"Precision": self.openmm_precision},
        )

        ##
        
        # === geoemtry optimization ===
        coarse_simulation.context.setPositions(self.modeller.getPositions())
        logging.info("skipping geometry optimization")
        # TODO: fix this
        # simulation.minimizeEnergy()
        # minimised_state = simulation.context.getState(
        #     getPositions=True, getVelocities=True, getForces=True
        #     )
        # # save GO minimized system
        # with open(os.path.join(self.output_dir, f"minimised_system.pdb"), "w") as f:
        #         PDBFile.writeFile(
        #             self.modeller.topology, minimised_state.getPositions(), file=f
        #         )
        ##
        
        # === assmble context ===
        fine_contexts = AssembleContexts(self.fine_system, fine_integrators)
        
        coarse_simulation.context.setVelocitiesToTemperature(self.temperature, seed)

        init_pos, init_vel = getPosVel(coarse_simulation)
        # N = NumCoarseSteps + 1
        # sol_x_init and sol_x_current is always not updated!!! 
        # It should be set directly from previous simumlations
        sol_x_init = [np.array([np.zeros(3) for _ in range(init_pos.shape[0])])] * N
        sol_v_init = [np.array([np.zeros(3) for _ in range(init_vel.shape[0])])] * N

        sol_x_init[0] = init_pos
        sol_v_init[0] = init_vel

        _G_L_init = np.random.normal(0, 1, (equilibrate_steps, npos, 3))
        G_L_init = [mat2vecvec(_G_L_init[k, :, :]) for k in range(equilibrate_steps)]
        
        total_dof = getdof(system=self.coarse_system)

        # equilibrate the system
        print("=== equilibration ===")
        #TODO: optimize this
        tmp_eqm_save = []
        for n in tqdm(range(equilibrate_steps)):
            integrator.setPerDofVariableByName("g1", G_L_init[n])
            coarse_simulation.step(1)
            state = coarse_simulation.context.getState(getEnergy=True)
            if n % 100 == 0:
                print(f"{n} equilibration temperature : ", (2*state.getKineticEnergy()/(total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin))
            tmp_eqm_save.append((2*state.getKineticEnergy()/(total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin))


        print("=== equilibration done ===")
        state = coarse_simulation.context.getState(getEnergy=True)
        ("Final temperature: ", (2*state.getKineticEnergy() / (total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin))


        # === what is written belows would be parareal ===
        current_positions, current_velocity = getPosVel(coarse_simulation)
        # saving initial position
        all_sol_x_array = [] + [current_positions,]
        all_sol_v_array = [] + [current_velocity,]
        all_temp_array = []
        all_delta_array = []
        
        # TODO: wrap this below into a function, this is so messy right now but easier for me to debug
        for kk in range(NumParaReal):
            print(f"Resolving T =: {kk * NumCoarseSteps} to {(kk + 1) * NumCoarseSteps} ...")
            
            # init 
            sol_x_init = [np.array([np.zeros(3) for _ in range(init_pos.shape[0])])] * N
            sol_v_init = [np.array([np.zeros(3) for _ in range(init_vel.shape[0])])] * N
            _G_L = np.random.normal(0, 1, (fine_steps_total, npos, 3))
            G_L = [mat2vecvec(_G_L[k, :, :]) for k in range(fine_steps_total)]
            
            # # TODO: clean up this loop
            # if kk > 1:                
            #     print(" ==== check norm === ")
            #     sol_x_init[0] = sol_x_current[-1]
            #     sol_v_init[0] = sol_v_current[-1]
            #     current_positions, current_velocity = getPosVel(coarse_simulation)
            # else:
            #     current_positions, current_velocity = getPosVel(coarse_simulation)
            #     sol_x_init[0] = current_positions
            #     sol_v_init[0] = current_velocity
            
            # the last position and velocity
            current_positions, current_velocity = getPosVel(coarse_simulation)
            sol_x_init[0] = current_positions
            sol_v_init[0] = current_velocity


            # coarse propogator initialization
            # TODO: optimize this
            print("=== coarse propogator init ===")
            for n in tqdm(range(NumCoarseSteps)):
                integrator.setPerDofVariableByName("g1", G_L[n * NumFineSteps])
                coarse_simulation.step(1)
                current_positions, current_velocity = getPosVel(coarse_simulation)
                sol_x_init[n+1] = current_positions
                sol_v_init[n+1] = current_velocity

            # set current
            sol_x_current = copy.deepcopy(sol_x_init)
            sol_v_current = copy.deepcopy(sol_v_init)
            temperature_array = [0] * N
            delta_array = []

            # set cost
            Cost = 0
            delta_x = delta_expl * 0.99
            delta_v = delta_expl * 0.99
            N_expl = 0

            
            # reinit variables
            N_init = 0
            N = NumCoarseSteps + 1
            N_final = copy.deepcopy(N)            
            
            end_flag = False
            max_nsteps = 200

            # set gaussian variable for different integrators here
            for (n, _integrator) in enumerate(fine_integrators):
                for j in range(NumFineSteps):
                    _integrator.setPerDofVariableByName("g%d" % j, G_L[n * NumFineSteps + j])
            print("Done setting gaussian variable")
            
            ##

            # adaptive PARAEAL iterations
            while N_init < N:
                while (delta_x > delta_conv_x or delta_v > delta_conv_v) and delta_x < delta_expl and delta_v < delta_expl:
                    # define prev as current
                    sol_x_prev = copy.deepcopy(sol_x_current)
                    sol_v_prev = copy.deepcopy(sol_v_current)

                    # === compute jump in parallel ===
                    # fine integrators in parallel: F_{deltat}(x^{prev}_{n})
                    # use fine_position, fine_velocity = getPosVel(fine_contexts[n])
                    # to get F_{deltat}(x^{prev}_{n})

                    for (idx, _context) in enumerate(fine_contexts):
                        _context.setPositions(sol_x_prev[idx])
                        _context.setVelocities(sol_v_prev[idx])
                    
                    print("Finish setting context positions")
                    
                    mytime = time.time()
                    
                    # === serial === TODO: this should be in parallel
                    for (n, _integrator) in enumerate(fine_integrators):
                        _integrator.setGlobalVariableByName("fine_iter", 0)
                        _integrator.step(NumFineSteps)
                        
                    print("Time for fine steps: ", time.time() - mytime)


                    # calculate the jump
                    #jump_x = [np.array([np.zeros(3) for _ in range(npos)])] * NumCoarseSteps
                    #jump_v = [np.array([np.zeros(3) for _ in range(npos)])] * NumCoarseSteps
                    
                    # TODO: adaptive to device later
                    jump_x = [torch.zeros((npos, 3), device = 'cuda', dtype = torch.float64)] * NumCoarseSteps
                    jump_v = [torch.zeros((npos, 3), device = 'cuda', dtype = torch.float64)] * NumCoarseSteps

                    print("start to calculate jump")
                    for n in range(N_init, N_final - 1):
                        coarse_simulation.context.setPositions(sol_x_prev[n])
                        coarse_simulation.context.setVelocities(sol_v_prev[n])
                        integrator.setPerDofVariableByName("g1", G_L[n * NumFineSteps])
                        coarse_simulation.step(1)
                        F_Dt_x_prev, F_Dt_v_prev = getPosVel_torch(fine_contexts[n])
                        C_Dt_x_prev, C_Dt_v_prev = getPosVel_torch(coarse_simulation)
                        jump_x[n] = F_Dt_x_prev - C_Dt_x_prev
                        jump_v[n] = F_Dt_v_prev - C_Dt_v_prev

                    Cost += 1
                    print("Cost: ", Cost)
                    if Cost > max_nsteps:
                        break
                    for n in range(N_init, N_final - 1):
                        # update parareal solution with jump and coarse step
                        coarse_simulation.context.setPositions(sol_x_current[n])
                        coarse_simulation.context.setVelocities(sol_v_current[n])
                        state = coarse_simulation.context.getState(getEnergy=True)
                        temperature_array[n] = (2*state.getKineticEnergy()/(total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin)
                        
                        integrator.setPerDofVariableByName("g1", G_L[n * NumFineSteps])
                        coarse_simulation.step(1)
                        C_Dt_x_current, C_Dt_v_current = getPosVel_torch(coarse_simulation.context)
                        # what writes into sol_x_current should always be np.array, or else seg fault for setPositions occur
                        # the "+" is done on GPU with torch.tensor and then we use array2np and write back to CPU
                        sol_x_current[n+1] = array2np(C_Dt_x_current + jump_x[n])
                        sol_v_current[n+1] = array2np(C_Dt_v_current + jump_v[n])

                        # compute relative error from parareal solutions
                        # TODO: we compute the error with respect to position only... does it affect the dynamics?
                        delta_x = parareal_error(N_init, n + 1, sol_x_prev, sol_x_current)
                        delta_v = parareal_error(N_init, n + 1, sol_v_prev, sol_v_current)
                        
                        if delta_x > delta_expl or delta_v > delta_expl:
                            N_expl = n + 1
                            break
                    state = coarse_simulation.context.getState(getEnergy=True)
                    temperature_array[N_final - 1] = (2*state.getKineticEnergy()/(total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin)
                    # setting position for the last step
                    coarse_simulation.context.setPositions(sol_x_current[N_final - 1])
                    coarse_simulation.context.setVelocities(sol_v_current[N_final - 1])
                        
                    print("N_init, N_final: ", N_init, N_final)
                    print("delta_x: ", delta_x)
                    print("delta_v: ", delta_v)
                    delta_array.append((delta_x, delta_v))
                # end while
                print("end while")
                
                if end_flag:
                    break
                
                # if explode, we set N_final to smaller interval and consider that smaller interval first
                # this is the adaptive part
                if delta_x > delta_expl or delta_v > delta_expl:
                    N_final = N_expl - 1
                else:
                    # this loop is only useful if delta > delta_expl previously, and N_final < N
                    # then after the previous slab has converged we set N_init = N_final and proceed
                    N_init = N_final
                    # update something here
                    for n in range(N_init, N - 1):
                        print("I should not get into this loop if N_init = 0 and N_final = N")
                        coarse_simulation.context.setPositions(sol_x_current[n])
                        coarse_simulation.context.setVelocities(sol_v_current[n])
                        integrator.setPerDofVariableByName("g1", G_L[n * NumFineSteps])
                        coarse_simulation.step(1)
                        sol_x_current[n+1], sol_v_current[n+1] = getPosVel(fine_contexts[n])

                    # setting the position for the final step separataly
                    coarse_simulation.context.setPositions(sol_x_current[N-1])
                    coarse_simulation.context.setVelocities(sol_v_current[N-1])
                    N_final = N
                state = coarse_simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
                delta_x = (delta_conv_x + delta_expl) / 2
                delta_v = (delta_conv_v + delta_expl) / 2
                print("Final temperature: ", (2*state.getKineticEnergy()/(total_dof*MOLAR_GAS_CONSTANT_R)).value_in_unit(kelvin))
            # end while
            # this should be done inside the forloop, but just to make sure it works fine
            coarse_simulation.context.setPositions(sol_x_current[N-1])
            coarse_simulation.context.setVelocities(sol_v_current[N-1])
            
            # merging arrays together for saving result
            # TODO: make this flush to a npy? otherwise there are memory problem?
            # the initial position is not saved
            all_sol_x_array = all_sol_x_array + sol_x_current[1:]
            all_sol_v_array = all_sol_v_array + sol_v_current[1:]
            all_temp_array = all_temp_array + temperature_array[1:]
            all_delta_array = all_delta_array + delta_array
        # save to numpy array for visualization
        np.save(os.path.join(self.output_dir, "sol_x_current.npy"), np.array(all_sol_x_array, dtype=object), allow_pickle=True)
        np.save(os.path.join(self.output_dir, "sol_v_current.npy"), np.array(all_sol_v_array, dtype=object), allow_pickle=True)
        np.savetxt(os.path.join(self.output_dir, "temperature_array.txt"), all_temp_array)
        np.save(os.path.join(self.output_dir, "delta_array.npy"), all_delta_array)
        np.savetxt(os.path.join(self.output_dir, "tmp_eqm_save.txt"), tmp_eqm_save)
        
        return None 

    def decouple_long_range(self, system: System, solute_indices: List) -> System:
        """Create an alchemically modified system with the lambda parameters to decouple the steric and electrostatic components of the forces according to their respective lambda parameters

        :param System system: the openMM system to test
        :param List solute_indices: the list of indices to treat as the alchemical region (i.e. the ligand to be decoupled from solvent)
        :return System: Alchemically modified version of the system with additional lambda parameters for the
        """
        factory = alchemy.AbsoluteAlchemicalFactory(alchemical_pme_treatment="exact")

        alchemical_region = alchemy.AlchemicalRegion(
            alchemical_atoms=solute_indices,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
        )
        alchemical_system = factory.create_alchemical_system(system, alchemical_region)

        return alchemical_system

    def run_neq_switching(
        self,
        steps: int,
        interval: int,
        output_file: str,
        restart: bool,
        direction: str = "forward",
    ) -> List[float]:
        """Compute the protocol work performed by switching from the MM description to the MM/ML through lambda_interpolate

        Ideally this will take a series of snapshots, probably as a pdb file, and run the switching

        :param int steps: number of steps in non-equilibrium switching simulation
        :param int interval: reporterInterval
        :return float: protocol work from the integrator
        """

        if direction not in ["forward", "reverse"]:
            raise ValueError("direction must be either forward or reverse")

        alchemical_functions = (
            {"lambda_interpolate": "lambda"}
            if direction == "forward"
            else {"lambda_interpolate": "1 - lambda"}
        )

        # steps = int(switching_time / self.timestep.value_in_unit(picosecond))
        logger.info("Running NEQ switching for {} steps".format(steps))
        # input file contains a trajectory of snapshots for which we need the work value associated with the switching
        # positions = self.neq_simulations_positions[positions_idx]
        # output_file = os.path.join(self.output_dir, f"neq_{direction}_{positions_idx}.pdb")
        # restart=False

        # # prepare a set of positions to run the switching simulation for
        # work_vals = []
        # for pos in positions:

        integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions=alchemical_functions,
            nsteps_neq=steps,
            temperature=self.temperature,
            collision_rate=self.friction_coeff,
            timestep=self.timestep,
            measure_shadow_work=False,
        )

        simulation = Simulation(
            self.modeller.topology,
            self.system,
            integrator,
        )
        simulation.context.setPositions(self.modeller.getPositions())

        # set velocities to temperature
        simulation.context.setVelocitiesToTemperature(self.temperature)

        # simulation.minimizeEnergy()

        reporter = StateDataReporter(
            file=sys.stdout,
            reportInterval=interval,
            step=True,
            time=True,
            totalEnergy=True,
            potentialEnergy=True,
            density=True,
            volume=True,
            temperature=True,
            speed=True,
            progress=True,
            totalSteps=steps,
            remainingTime=True,
        )
        simulation.reporters.append(reporter)
        # keep periodic box off to make quick visualisation easier
        simulation.reporters.append(
            PDBReporter(
                file=os.path.join(self.output_dir, output_file),
                reportInterval=interval,
                enforcePeriodicBox=False if self.unwrap else True,
            )
        )
        # we need this to hold the box vectors for NPT simulations
        # netcdf_reporter = NetCDFReporter(
        #     file=os.path.join(self.output_dir, output_file[:-4] + ".nc"),
        #     reportInterval=interval,
        # )
        # simulation.reporters.append(netcdf_reporter)
        dcd_reporter = DCDReporter(
            file=os.path.join(self.output_dir, "output.dcd"),
            reportInterval=interval,
            append=restart,
            enforcePeriodicBox=False if self.unwrap else True,
        )
        simulation.reporters.append(dcd_reporter)
        # hdf5_reporter = HDF5Reporter(
        #     file=os.path.join(self.output_dir, output_file[:-4] + ".h5"),
        #     reportInterval=interval,
        #     velocities=True,
        # )
        # simulation.reporters.append(hdf5_reporter)
        # Add an extra hash to any existing checkpoint files
        checkpoint_files = [f for f in os.listdir(self.output_dir) if f.endswith("#")]
        for file in checkpoint_files:
            os.rename(
                os.path.join(self.output_dir, file),
                os.path.join(self.output_dir, f"{file}#"),
            )

        checkpoint_filepath = os.path.join(self.output_dir, output_file[:-4] + ".chk")
        # backup the existing checkpoint file
        if os.path.isfile(checkpoint_filepath):
            os.rename(checkpoint_filepath, checkpoint_filepath + "#")
        checkpoint_reporter = CheckpointReporter(
            file=checkpoint_filepath, reportInterval=interval
        )
        simulation.reporters.append(checkpoint_reporter)

        # We need to take the final state
        simulation.step(steps)
        protocol_work = integrator.get_total_work(dimensionless=True)
        print(f"Protocol work: {protocol_work}")
        return protocol_work


class MixedSystem(MACESystemBase):
    forcefields: List[str]
    padding: float
    ionicStrength: float
    nonbondedCutoff: float
    resname: str
    nnpify_type: str
    mixed_system: System
    minimise: bool
    water_model: str

    def __init__(
        self,
        file: str,
        ml_mol: str,
        model_path: str,
        resname: str,
        nnpify_type: str,
        potential: str,
        nl: str,
        max_n_pairs: int,
        minimiser: str,
        output_dir: str,
        padding: float = 1.2,
        shape: str = "cube",
        ionicStrength: float = 0.15,
        forcefields: List[str] = [
            "amber/protein.ff14SB.xml",
            "amber14/DNA.OL15.xml",
            "amber/tip3p_standard.xml",
        ],
        nonbondedCutoff: float = 1.0,
        temperature: float = 298,
        dtype: torch.dtype = torch.float64,
        decouple: bool = False,
        interpolate: bool = False,
        mm_only: bool = False,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        water_model: str = "tip3p",
        pressure: Optional[float] = None,
        cv1: Optional[str] = None,
        cv2: Optional[str] = None,
        write_gmx: bool = False,
        remove_cmm=False,
        unwrap=False,
        set_temperature=False,
    ) -> None:
        super().__init__(
            file=file,
            model_path=model_path,
            potential=potential,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            friction_coeff=friction_coeff,
            timestep=timestep,
            smff=smff,
            nl=nl,
            max_n_pairs=max_n_pairs,
            minimiser=minimiser,
            mm_only=mm_only,
            remove_cmm=remove_cmm,
            unwrap=unwrap,
            set_temperature=set_temperature,
        )

        self.forcefields = forcefields
        self.padding = padding
        self.shape = shape
        self.ionicStrength = ionicStrength
        self.nonbondedCutoff = nonbondedCutoff
        self.resname = resname
        self.nnpify_type = nnpify_type
        self.cv1 = cv1
        self.cv2 = cv2
        self.water_model = water_model
        self.decouple = decouple
        self.interpolate = interpolate
        self.write_gmx = write_gmx

        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        # created the hybrid system
        self.create_system(
            file=file,
            ml_mol=ml_mol,
            model_path=model_path,
        )

    def create_system(
        self,
        file: str,
        model_path: str,
        ml_mol: str,
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        if ml_mol is not None:
            atoms, molecule = self.initialize_ase_atoms(ml_mol)
        else:
            atoms, molecule = None, None
        # set the default topology to that of the ml molecule, this will get overwritten below

        # Handle a complex, passed as a pdb file
        if file.endswith(".pdb"):
            input_file = PDBFile(file)
            topology = input_file.getTopology()

            self.modeller = Modeller(input_file.topology, input_file.positions)
            logger.info(
                f"Initialized topology with {len(input_file.positions)} positions"
            )

        # Handle a small molecule/small periodic system, passed as an sdf or xyz
        # this should also handle generating smirnoff parameters for something like an octa-acid, where this is still to be handled by the MM forcefield, but needs parameters generated
        elif file.endswith(".sdf") or file.endswith(".xyz"):
            # handle the case where the receptor and ligand are both passed as different sdf files:
            if ml_mol != file:
                logger.info("Combining and parametrising 2 sdf files...")
                # load the receptor
                receptor_as_molecule = Molecule.from_file(file)

                # create modeller from this
                self.modeller = Modeller(
                    receptor_as_molecule.to_topology().to_openmm(),
                    get_xyz_from_mol(receptor_as_molecule.to_rdkit()) / 10,
                )
                # combine with modeller for the ml_mol
                ml_mol_modeller = Modeller(
                    molecule.to_topology().to_openmm(),
                    get_xyz_from_mol(molecule.to_rdkit()) / 10,
                )

                self.modeller.add(ml_mol_modeller.topology, ml_mol_modeller.positions)
                # send both to the forcefield initializer
                molecule = [molecule, receptor_as_molecule]

            else:
                input_file = molecule
                topology = molecule.to_topology().to_openmm()
                # Hold positions in nanometers
                positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

                logger.info(f"Initialized topology with {positions.shape} positions")

                self.modeller = Modeller(topology, positions)

        forcefield = initialize_mm_forcefield(
            molecule=molecule, forcefields=self.forcefields, smff=self.SM_FF
        )
        if self.write_gmx:
            from openff.interchange import Interchange

            interchange = Interchange.from_smirnoff(
                topology=molecule.to_topology(), force_field=ForceField(self.SM_FF)
            )
            interchange.to_top(os.path.join(self.output_dir, "topol.top"))
            interchange.to_gro(os.path.join(self.output_dir, "conf.gro"))
        if self.padding > 0:
            logger.info(f"Adding {self.shape} solvent box")
            if "tip4p" in self.water_model:
                self.modeller.addExtraParticles(forcefield)
            self.modeller.addSolvent(
                forcefield,
                model=self.water_model,
                padding=self.padding * nanometers,
                boxShape=self.shape,
                ionicStrength=self.ionicStrength * molar,
                neutralize=False,
            )

            omm_box_vecs = self.modeller.topology.getPeriodicBoxVectors()
            # ensure atoms object has boxvectors taken from the PDB file
            if atoms is not None:
                atoms.set_cell(
                    [
                        omm_box_vecs[0][0].value_in_unit(angstrom),
                        omm_box_vecs[1][1].value_in_unit(angstrom),
                        omm_box_vecs[2][2].value_in_unit(angstrom),
                    ]
                )
        # else:
        # this should be a large enough box
        # run a non-periodic simulation
        # self.modeller.topology.setPeriodicBoxVectors([[5, 0, 0], [0, 5, 0], [0, 0, 5]])

        system = forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=PME
            if self.modeller.topology.getPeriodicBoxVectors() is not None
            else CutoffNonPeriodic,
            nonbondedCutoff=self.nonbondedCutoff * nanometers,
            constraints=None if "unconstrained" in self.SM_FF else HBonds,
        )

        # write the final prepared system to disk
        with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
            PDBFile.writeFile(
                self.modeller.topology, self.modeller.getPositions(), file=f
            )

        # if self.write_gmx:
        #     # write the openmm system to gromacs top/gro with parmed
        #     from parmed.openmm import load_topology

        #     parmed_structure = load_topology(self.modeller.topology, system)
        #     parmed_structure.save(os.path.join(self.output_dir, "topol_full.top"), overwrite=True)
        #     parmed_structure.save(os.path.join(self.output_dir, "conf_full.gro"), overwrite=True)
        #     raise KeyboardInterrupt

        if not self.decouple:
            if self.mm_only:
                logger.info("Creating MM system")
                self.system = system
            else:
                logger.debug("Creating hybrid system")
                self.system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_id=self.resname,
                    model_path=model_path,
                    nnp_potential=self.potential,
                    nnpify_type=self.nnpify_type,
                    atoms_obj=atoms,
                    interpolate=self.interpolate,
                    filename=model_path,
                    dtype=self.dtype,
                    nl=self.nl,
                    max_n_pairs=self.max_n_pairs
                ).mixed_system

            # optionally, add the alchemical customCVForce for the nonbonded interactions to run ABFE edges
        else:
            if not self.mm_only:
                # TODO: implement decoupled system for VdW/coulomb forces
                logger.info("Creating decoupled system")
                self.system = MixedSystemConstructor(
                    system=system,
                    topology=self.modeller.topology,
                    nnpify_type=self.nnpify_type,
                    nnpify_id=self.resname,
                    nnp_potential=self.potential,
                    model_path=model_path,
                    # cannot have the lambda parameter for this as well as the electrostatics/sterics being decoupled
                    interpolate=False,
                    atoms_obj=atoms,
                    filename=model_path,
                    dtype=self.dtype,
                    nl=self.nl,
                ).mixed_system

            self.system = self.decouple_long_range(
                system,
                solute_indices=get_atoms_from_resname(
                    self.modeller.topology, self.resname, self.nnpify_type
                ),
            )


class PureSystem(MACESystemBase):
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    boxsize: Optional[int]

    def __init__(
        self,
        ml_mol: str,
        model_path: str,
        potential: str,
        output_dir: str,
        temperature: float,
        nl: str,
        max_n_pairs: int,
        minimiser: str,
        file: Optional[str] = None,
        boxsize: Optional[int] = None,
        pressure: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        remove_cmm: bool = False,
        unwrap: bool = False,
        set_temperature: bool = False,
    ) -> None:
        super().__init__(
            # if file is None, we don't need  to create a topology, so we can pass the ml_mol
            file=ml_mol if file is None else file,
            model_path=model_path,
            potential=potential,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            nl=nl,
            max_n_pairs=max_n_pairs,
            friction_coeff=friction_coeff,
            timestep=timestep,
            smff=smff,
            minimiser=minimiser,
            remove_cmm=remove_cmm,
            unwrap=unwrap,
            set_temperature=set_temperature,
        )
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.boxsize = boxsize

        self.create_system(ml_mol=ml_mol, model_path=model_path)

    def create_system(
        self,
        ml_mol: str,
        model_path: str,
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        # initialize the ase atoms for MACE
        atoms = read(ml_mol)
        if self.minimiser == "ase":
            # ensure the model was saved on the GPU
            tmp_model = torch.load(model_path, map_location="cpu")
            _, tmp_path = mkstemp(suffix=".pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("debug:", device)
            tmp_model = tmp_model.to(device)
            torch.save(tmp_model, tmp_path)
            calc = MACECalculator(
                model_paths=tmp_path,
                device="cuda",
                default_dtype=self.dtype.__str__().split(".")[1],
            )
            atoms.set_calculator(calc)
            # minimise the system with ase
            logger.info("Minimising with ASE...")
            opt = LBFGS(atoms)
            opt.run(fmax=0.2)
            os.remove(tmp_path)

        # write out minimised system
        write(os.path.join(self.output_dir, "minimised.xyz"), atoms)

        if ml_mol.endswith(".xyz"):
            pos = atoms.get_positions() / 10
            box_vectors = atoms.get_cell() / 10
            # canonicalise
            if max(atoms.get_cell().cellpar()[:3]) > 0:
                box_vectors = reducePeriodicBoxVectors(box_vectors)
            logger.info(f"Using reduced periodic box vectors {box_vectors}")
            elements = atoms.get_chemical_symbols()

            # Create a topology object
            topology = Topology()

            # Add atoms to the topology
            chain = topology.addChain()
            res = topology.addResidue("mace_system", chain)
            for i, (element, position) in enumerate(zip(elements, pos)):
                e = Element.getBySymbol(element)
                topology.addAtom(str(i), e, res)
            # if there is a periodic box specified add it to the Topology
            if max(atoms.get_cell().cellpar()[:3]) > 0:
                topology.setPeriodicBoxVectors(vectors=box_vectors)
            # if there is a periodic box on the pdb file and not the xyz, use that
        
            # load the pdbfile
            # pdb_top = PDBFile(self.file)
            # # extract the boxvectors
            # box_vectors = pdb_top.topology.getPeriodicBoxVectors()
            # topology.setPeriodicBoxVectors(box_vectors)
            #
            # logger.info(f"Initialized topology with {pos.shape} positions")

            self.modeller = Modeller(topology, pos)

        elif ml_mol.endswith(".sdf"):
            molecule = Molecule.from_file(ml_mol)
            # input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            # Manually attach periodic box if requested|
            if self.boxsize is not None:
                boxvecs = np.eye(3, 3) * self.boxsize
                topology.setPeriodicBoxVectors(boxvecs)

            logger.info(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)
        elif ml_mol.endswith(".pdb"):
            # create a modeller from the pdb file
            input_file = PDBFile(self.file)
            topology = input_file.getTopology()

            self.modeller = Modeller(topology, input_file.positions)


            logger.info(f"Parased box vectors {self.modeller.topology.getPeriodicBoxVectors()} from pdb file")

        # write the prepared system to pd bfile
            with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
                PDBFile.writeFile(
                    self.modeller.topology, self.modeller.getPositions(), file=f
                )

        else:
            raise NotImplementedError

        # TODO: allow here to construct other potentials, maybe add an extra argument etc.
        # the GO can always be done by mace, it doesn't matter. 
        ml_potential = MLPotential(self.potential, model_path=model_path)
        self.system = ml_potential.createSystem(topology,
                                                dtype=self.dtype,
                                                nl=self.nl, 
                                                max_n_pairs=self.max_n_pairs, 
                                                ) 

        # if pressure is not None:
        #     logger.info(
        #         f"Pressure will be maintained at {pressure} bar with MC barostat"
        #     )
        #     barostat = MonteCarloBarostat(pressure * bar, self.temperature * kelvin)
        #     # barostat.setFrequency(25)  25 timestep is the default
        #     self.system.addForce(barostat)

class PararealSystem(MACESystemBase):
    potential: str
    temperature: float
    friction_coeff: float
    timestep: float
    dtype: torch.dtype
    output_dir: str
    openmm_precision: str
    SM_FF: str
    modeller: Modeller
    boxsize: Optional[int]

    def __init__(
        self,
        ml_mol: str,
        model_path: str,
        potential: str,
        output_dir: str,
        temperature: float,
        nl: str,
        max_n_pairs: int,
        minimiser: str,
        # my arguments
        coarse_potential_path: str,
        nonbounded_method,
        coarse_mace = False,
        ##
        file: Optional[str] = None,
        boxsize: Optional[int] = None,
        pressure: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
        friction_coeff: float = 1.0,
        timestep: float = 1.0,
        smff: str = "1.0",
        remove_cmm: bool = False,
        unwrap: bool = False,
        set_temperature: bool = False,
    ) -> None:
        super().__init__(
            # if file is None, we don't need  to create a topology, so we can pass the ml_mol
            file=ml_mol if file is None else file,
            model_path=model_path,
            potential=potential,
            output_dir=output_dir,
            temperature=temperature,
            pressure=pressure,
            dtype=dtype,
            nl=nl,
            max_n_pairs=max_n_pairs,
            friction_coeff=friction_coeff,
            timestep=timestep,
            smff=smff,
            minimiser=minimiser,
            remove_cmm=remove_cmm,
            unwrap=unwrap,
            set_temperature=set_temperature,
        )
        logger.debug(f"OpenMM will use {self.openmm_precision} precision")

        self.boxsize = boxsize

        self.create_system(ml_mol=ml_mol, model_path=model_path, coarse_potential_path = coarse_potential_path,
                           nonbounded_method = nonbounded_method, coarse_mace = coarse_mace)

    def create_system(
        self,
        ml_mol: str,
        model_path: str,
        coarse_potential_path: str, # TODO: currently this must be coming from openmm, generalize later
        nonbounded_method,
        coarse_mace
    ) -> None:
        """Creates the mixed system from a purely mm system

        :param str file: input pdb file
        :param str model_path: path to the mace model
        :return Tuple[System, Modeller]: return mixed system and the modeller for topology + position access by downstream methods
        """
        # initialize the ase atoms for MACE
        atoms = read(ml_mol)
        if self.minimiser == "ase":
            # ensure the model was saved on the GPU
            tmp_model = torch.load(model_path, map_location="cpu")
            _, tmp_path = mkstemp(suffix=".pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("debug:", device)
            tmp_model = tmp_model.to(device)
            torch.save(tmp_model, tmp_path)
            calc = MACECalculator(
                model_paths=tmp_path,
                device="cuda",
                default_dtype=self.dtype.__str__().split(".")[1],
            )
            atoms.set_calculator(calc)
            # minimise the system with ase
            logger.info("Minimising with ASE...")
            opt = LBFGS(atoms)
            opt.run(fmax=0.2)
            os.remove(tmp_path)

        # write out minimised system
        write(os.path.join(self.output_dir, "minimised.xyz"), atoms)

        if ml_mol.endswith(".xyz"):
            pos = atoms.get_positions() / 10
            box_vectors = atoms.get_cell() / 10
            # canonicalise
            if max(atoms.get_cell().cellpar()[:3]) > 0:
                box_vectors = reducePeriodicBoxVectors(box_vectors)
            logger.info(f"Using reduced periodic box vectors {box_vectors}")
            elements = atoms.get_chemical_symbols()

            # Create a topology object
            topology = Topology()

            # Add atoms to the topology
            chain = topology.addChain()
            res = topology.addResidue("mace_system", chain)
            for i, (element, position) in enumerate(zip(elements, pos)):
                e = Element.getBySymbol(element)
                topology.addAtom(str(i), e, res)
            # if there is a periodic box specified add it to the Topology
            if max(atoms.get_cell().cellpar()[:3]) > 0:
                topology.setPeriodicBoxVectors(vectors=box_vectors)
            # if there is a periodic box on the pdb file and not the xyz, use that
        
            # load the pdbfile
            # pdb_top = PDBFile(self.file)
            # # extract the boxvectors
            # box_vectors = pdb_top.topology.getPeriodicBoxVectors()
            # topology.setPeriodicBoxVectors(box_vectors)
            #
            # logger.info(f"Initialized topology with {pos.shape} positions")

            self.modeller = Modeller(topology, pos)

        elif ml_mol.endswith(".sdf"):
            molecule = Molecule.from_file(ml_mol)
            # input_file = molecule
            topology = molecule.to_topology().to_openmm()
            # Hold positions in nanometers
            positions = get_xyz_from_mol(molecule.to_rdkit()) / 10

            # Manually attach periodic box if requested|
            if self.boxsize is not None:
                boxvecs = np.eye(3, 3) * self.boxsize
                topology.setPeriodicBoxVectors(boxvecs)

            logger.info(f"Initialized topology with {positions.shape} positions")

            self.modeller = Modeller(topology, positions)
        elif ml_mol.endswith(".pdb"):
            # create a modeller from the pdb file
            input_file = PDBFile(self.file)
            topology = input_file.getTopology()

            self.modeller = Modeller(topology, input_file.positions)


            logger.info(f"Parased box vectors {self.modeller.topology.getPeriodicBoxVectors()} from pdb file")

        # write the prepared system to pd bfile
            with open(os.path.join(self.output_dir, "prepared_system.pdb"), "w") as f:
                PDBFile.writeFile(
                    self.modeller.topology, self.modeller.getPositions(), file=f
                )

        else:
            raise NotImplementedError

        # TODO: allow here to construct other potentials, maybe add an extra argument etc.
        # the GO can always be done by mace, it doesn't matter.
        fine_potential = MLPotential(self.potential, model_path=model_path)
        coarse_potential = app.ForceField(coarse_potential_path)
        
        self.fine_system = fine_potential.createSystem(topology,
                                                nonbondedMethod=nonbounded_method,
                                                dtype=self.dtype,
                                                nl=self.nl, 
                                                max_n_pairs=self.max_n_pairs, 
                                                )
        
        if coarse_mace:
            print("running with coarse mace propogator - the emperical ff is not used")

        self.coarse_system = self.coarse_system = fine_potential.createSystem(topology,
                                                nonbondedMethod=nonbounded_method,
                                                dtype=self.dtype,
                                                nl=self.nl, 
                                                max_n_pairs=self.max_n_pairs, 
                                                ) if coarse_mace else coarse_potential.createSystem(topology, nonbondedMethod=nonbounded_method,)
