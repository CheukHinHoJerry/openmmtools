from multipledispatch import dispatch
from threading import Thread
from multiprocessing.context import Process
from tqdm import tqdm

import openmm
import multiprocessing
import torch
import numpy

@dispatch(openmm.app.simulation.Simulation)
def getPosVel(_simulation):
    pos_vel = _simulation.context.getState(getPositions=True, getVelocities=True)
    current_positions = pos_vel.getPositions(asNumpy=True)
    current_velocity = pos_vel.getVelocities(asNumpy=True)
    return current_positions, current_velocity

@dispatch(openmm.openmm.Context)
def getPosVel(_context):
    pos_vel = _context.getState(getPositions=True, getVelocities=True)
    current_positions = pos_vel.getPositions(asNumpy=True) # a vector of 3d numpy array
    current_velocity = pos_vel.getVelocities(asNumpy=True)
    return current_positions, current_velocity

def np2torch(np_array, device = 'cuda'):
    return torch.asarray(np_array.value_in_unit(np_array.unit), device = device) * np_array.unit

@dispatch(openmm.unit.quantity.Quantity)
def array2np(quan_array):
    if isinstance(quan_array.value_in_unit(quan_array.unit), torch.Tensor):
        return quan_array.numpy(force = True) * quan_array.unit
    elif isinstance(quan_array.value_in_unit(quan_array.unit), numpy.ndarray):
        return quan_array
    else:
        TypeError("array2np functino get quan_array with value not equal to torch.Tensor or np array")

@dispatch(openmm.app.simulation.Simulation)
def getPosVel_torch(_simulation, device = 'cuda'):
    pos_vel = _simulation.context.getState(getPositions=True, getVelocities=True)
    current_positions = pos_vel.getPositions(asNumpy=True)
    current_velocity = pos_vel.getVelocities(asNumpy=True)
    return np2torch(current_positions, device = device), np2torch(current_velocity, device = device)

@dispatch(openmm.openmm.Context)
def getPosVel_torch(_context, device = 'cuda'):
    pos_vel = _context.getState(getPositions=True, getVelocities=True)
    current_positions = pos_vel.getPositions(asNumpy=True)
    current_velocity = pos_vel.getVelocities(asNumpy=True)
    return np2torch(current_positions, device = device), np2torch(current_velocity, device = device)

def PutContext(system, fi, _q, idx):
    _q[idx] = openmm.openmm.Context(system, fi)

def AssembleContexts(system, fine_integrators):
    fine_contexts = [None] * len(fine_integrators)
    procs = []
    print("assembling contextes")
    for (idx, fi) in enumerate(fine_integrators):
        proc = Thread(target=PutContext, args=(system, fi, fine_contexts, idx))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in tqdm(procs):
        proc.join()
    print("finish assmebling contexts")
    return fine_contexts

# converting matrix into vector of vector
def mat2vecvec(mat):
    npos = mat.shape[0]
    return [mat[t, :] for t in range(npos)]

       
def getdof(system):
    # Compute the number of degrees of freedom.
    dof = 0
    for i in range(system.getNumParticles()):
        if system.getParticleMass(i) > 0*openmm.unit.dalton:
            dof += 3
    for i in range(system.getNumConstraints()):
        p1, p2, distance = system.getConstraintParameters(i)
        if system.getParticleMass(p1) > 0*openmm.unit.dalton or system.getParticleMass(p2) > 0* openmm.unit.dalton:
            dof -= 1
    if any(type(system.getForce(i)) == openmm.CMMotionRemover for i in range(system.getNumForces())):
        dof -= 3
    return dof

def run_fine_steps(integrator, NumFineSteps):#, G_L):
    integrator.setGlobalVariableByName("fine_iter", 0)
    integrator.step(NumFineSteps)