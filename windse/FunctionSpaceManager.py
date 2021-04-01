"""
The FunctionSpaceManager contains all the different types of function
spaces required for solve multiple classes of problems.
"""

import __main__
import os

### Get the name of program importing this package ###
if hasattr(__main__,"__file__"):
    main_file = os.path.basename(__main__.__file__)
else:
    main_file = "ipython"
    
### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    from dolfin import *
    import time

    ### Import the cumulative parameters ###
    from windse import windse_parameters

    ### Check if we need dolfin_adjoint ###
    if windse_parameters.dolfin_adjoint:
        from dolfin_adjoint import *



# Sub domain for Periodic boundary condition
class SinglePeriodicBoundary(SubDomain):

    def __init__(self,a,b,axis=1):
        super(SinglePeriodicBoundary, self).__init__()
        self.axis   = axis # 0 for periodic x and 1 for periodic y
        self.target = a    # location of the primary boundary 
        self.offset = b-a  # the offset from the primary to the secondary 

    # returns true primary boundary 
    def inside(self, x, on_boundary):
        return near(x[self.axis],self.target) and on_boundary

    # this maps the secondary boundary to the primary boundary
    def map(self, x, y):
        y[:] = x[:] # Set all equal
        y[self.axis] = x[self.axis] - self.offset # Modify the value that is periodic

# class DoublePeriodicBoundary(SubDomain):


#     def __init__(self,x_range,y_range,spanwise_periodic,streamwise_periodic):
#         super(PeriodicBoundary, self).__init__()

#     # Left boundary is "target domain" 
#     def inside(self, x, on_boundary):
#         return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

#     # Map right boundary (H) to left boundary (G)
#     def map(self, x, y):
#         y[0] = x[0] - 1.0
#         y[1] = x[1]




class GenericFunctionSpace(object):
    def __init__(self,dom):
        self.params = windse_parameters
        self.fprint = self.params.fprint
        self.tag_output = self.params.tag_output
        self.debug_mode = self.params.debug_mode
        self.dim = dom.dim
        self.mesh = dom.mesh

        ### Update attributes based on params file ###
        for key, value in self.params["function_space"].items():
            setattr(self,key,value)
        self.turbine_method = self.params["wind_farm"]["turbine_method"]
        self.spanwise_periodic = self.params["boundary_conditions"]["spanwise_periodic"]

        if self.turbine_space == "Quadrature" and (self.turbine_degree != self.quadrature_degree):
            raise ValueError("When using the numpy representation with the 'Quadrature' space, the turbine degree and quadrature degree must be equal.")

    def SetupSubspaces(self):
        self.V = self.W.sub(0).collapse()
        self.Q = self.W.sub(1).collapse()
        self.V0 = self.V.sub(0).collapse() 
        self.V1 = self.V.sub(1).collapse()

        if self.dim == 3: 
            self.V2 = self.V.sub(2).collapse()
            self.VelocityAssigner = FunctionAssigner(self.V,[self.V0,self.V1,self.V2])
        else:
            self.VelocityAssigner = FunctionAssigner(self.V,[self.V0,self.V1])

        self.SolutionAssigner = FunctionAssigner(self.W,[self.V,self.Q])

        ### Create Function Spaces for numpy turbine force ###
        if self.turbine_method == "numpy":
            tf_V = VectorElement(self.turbine_space,self.mesh.ufl_cell(),degree=self.turbine_degree,quad_scheme="default")
            self.tf_V = FunctionSpace(self.mesh, tf_V, constrained_domain=pbc)
            self.tf_V0 = self.tf_V.sub(0).collapse() 
            self.fprint("Quadrature DOFS: {:d}".format(self.tf_V.dim()))

    def DebugOutput(self):
        if self.debug_mode:
            self.tag_output("velocity_dofs",self.V.dim())
            self.tag_output("pressure_dofs",self.Q.dim())
            self.tag_output("total_dofs",self.W.dim())


class LinearFunctionSpace(GenericFunctionSpace):
    """
    The LinearFunctionSpace is made up of a vector function space for velocity
    and a scaler space for pressure. Both spaces are "CG1" or Linear Lagrange elements.
    """
    def __init__(self,dom):
        super(LinearFunctionSpace, self).__init__(dom)

        ### Create the function space ###
        fs_start = time.time()
        self.fprint("Creating Function Space",special="header")

        if self.spanwise_periodic:
            self.fprint("Using Spanwise Periodic Boundaries")
            pbc = SinglePeriodicBoundary(dom.y_range[0],dom.y_range[1],axis=1)
            dom.excluded_boundaries = ["north","south"]
        else:
            pbc = None

        V = VectorElement('Lagrange', self.mesh.ufl_cell(), 1) 
        Q = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1)
        self.T = FunctionSpace(dom.mesh, TensorElement('Lagrange', dom.mesh.ufl_cell(), 1), constrained_domain=pbc)
        self.W = FunctionSpace(self.mesh, MixedElement([V,Q]), constrained_domain=pbc)

        self.SetupSubspaces()

        self.fprint("Velocity DOFS:   {:d}".format(self.V.dim()))
        self.fprint("Pressure DOFS:   {:d}".format(self.Q.dim()))
        self.fprint("Total DOFS:      {:d}".format(self.W.dim()))
        
        self.DebugOutput()

        fs_stop = time.time()
        self.fprint("Function Spaces Created: {:1.2f} s".format(fs_stop-fs_start),special="footer")



class TaylorHoodFunctionSpace(GenericFunctionSpace):
    """
    The TaylorHoodFunctionSpace is made up of a vector function space for velocity
    and a scalar space for pressure. The velocity function space is piecewise quadratic
    and the pressure function space is piecewise linear.
    """
    def __init__(self,dom):
        super(TaylorHoodFunctionSpace, self).__init__(dom)

        ### Create the function space ###
        fs_start = time.time()
        self.fprint("Creating Function Space",special="header")
        V = VectorElement('Lagrange', self.mesh.ufl_cell(), 2, constrained_domain=pbc) 
        Q = FiniteElement('Lagrange', self.mesh.ufl_cell(), 1, constrained_domain=pbc)
        self.W = FunctionSpace(self.mesh, MixedElement([V,Q]))

        self.SetupSubspaces()

        self.fprint("Velocity DOFS: {:d}".format(self.V.dim()))
        self.fprint("Pressure DOFS: {:d}".format(self.Q.dim()))
        self.fprint("Total DOFS:    {:d}".format(self.W.dim()))

        self.DebugOutput()

        fs_stop = time.time()
        self.fprint("Function Spaces Created: {:1.2f} s".format(fs_stop-fs_start),special="footer")

