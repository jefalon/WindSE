""" 
This is the init file for WindSE. It handle importing all the
submodules and initializing the parameters.
"""

from windse.ParameterManager import windse_parameters
def initialize(loc,updated_parameters=[]):
    """
    This function initializes all the submodules in WindSE.

    Args:
        loc (str): This string is the location of the .yaml parameters file.

    """

    windse_parameters.Load(loc,updated_parameters=updated_parameters)

    global BaseHeight, CalculateDiskTurbineForces, UpdateActuatorLineForce, RadialChordForce, Optimizer#, ReducedFunctional
    if windse_parameters["general"].get("dolfin_adjoint", False):
        from windse.dolfin_adjoint_helper import BaseHeight, CalculateDiskTurbineForces, UpdateActuatorLineForce, RadialChordForce#, ReducedFunctional
        from windse.OptimizationManager import Optimizer
    else:
        from windse.helper_functions import BaseHeight, CalculateDiskTurbineForces, UpdateActuatorLineForce, RadialChordForce
    
    global BoxDomain, CylinderDomain, CircleDomain, RectangleDomain, ImportedDomain, InterpolatedCylinderDomain, InterpolatedBoxDomain
    from windse.DomainManager import BoxDomain, CylinderDomain, CircleDomain, RectangleDomain, ImportedDomain, InterpolatedCylinderDomain, InterpolatedBoxDomain

    global GridWindFarm, RandomWindFarm, ImportedWindFarm
    from windse.WindFarmManager import GridWindFarm, RandomWindFarm, ImportedWindFarm

    global RefineMesh, WarpMesh
    from windse.RefinementManager import RefineMesh, WarpMesh

    global LinearFunctionSpace, TaylorHoodFunctionSpace
    from windse.FunctionSpaceManager import LinearFunctionSpace, TaylorHoodFunctionSpace

    global PowerInflow, UniformInflow, UniformInflowTurn, RotatingSineInflow, LogLayerInflow, TurbSimInflow
    from windse.BoundaryManager import PowerInflow, UniformInflow, UniformInflowTurn, RotatingSineInflow, LogLayerInflow, TurbSimInflow

    global StabilizedProblem, TaylorHoodProblem, UnsteadyProblem
    from windse.ProblemManager import StabilizedProblem, TaylorHoodProblem, UnsteadyProblem

    global SteadySolver, UnsteadySolver, MultiAngleSolver, TimeSeriesSolver
    from windse.SolverManager import SteadySolver, UnsteadySolver, MultiAngleSolver, TimeSeriesSolver
