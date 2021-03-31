""" 
The BoundaryManager submodule contains the classes required for 
defining the boundary conditions. 
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
    from firedrake import *
    import numpy as np

    ### Import the cumulative parameters ###
    from windse import windse_parameters

    ### Check if we need dolfin_adjoint ###
    if windse_parameters.dolfin_adjoint:
        from dolfin_adjoint import *  

    import math 
    from scipy.interpolate import RegularGridInterpolator


class GenericBoundary(object):
    def __init__(self,dom,fs,farm):
        self.params = windse_parameters
        self.dom = dom
        self.fs = fs
        self.farm = farm
        self.ig_first_save = True
        self.height_first_save = True
        self.fprint = self.params.fprint
        self.tag_output = self.params.tag_output
        self.debug_mode = self.params.debug_mode
        
        ### Update attributes based on params file ###
        for key, value in self.params["boundary_conditions"].items():
            setattr(self,key,value)

        ### get the height to apply the HH_vel ###
        if self.vel_height == "HH":
            self.vel_height = np.mean(farm.HH)
        if np.isnan(self.vel_height):
            raise ValueError("Hub Height not defined, likely and EmptyFarm. Please set boundary_conditions:vel_height in config yaml")

        ### Get solver parameters ###
        self.final_time = self.params["solver"]["final_time"]

        ### Define the zero function based on domain dimension ###
        self.zeros = Constant(dom.dim*(0.0,))
        self.zero  = Constant(0.0)

        ### Use custom boundary tags if provided ###
        if self.params.default_bc_names:
            self.boundary_names = self.dom.boundary_names
        if self.params.default_bc_types:
            self.boundary_types = self.dom.boundary_types

    def DebugOutput(self):
        if self.debug_mode:
            # Average of the x and y-velocities
            self.tag_output("min_x", np.min(self.ux.vector()[:]))
            self.tag_output("max_x", np.max(self.ux.vector()[:]))
            self.tag_output("avg_x", np.mean(self.ux.vector()[:]))
            self.tag_output("min_y", np.min(self.uy.vector()[:]))
            self.tag_output("max_y", np.max(self.uy.vector()[:]))
            self.tag_output("avg_y", np.mean(self.uy.vector()[:]))

            # If applicable, average of z-velocities
            if self.dom.dim == 3:
                self.tag_output("min_z", np.min(self.uz.vector()[:]))
                self.tag_output("max_z", np.max(self.uz.vector()[:]))
                self.tag_output("avg_z", np.mean(self.uz.vector()[:]))

            # Average of the pressures
            self.tag_output("min_p", np.min(self.bc_pressure.vector()[:]))
            self.tag_output("max_p", np.max(self.bc_pressure.vector()[:]))
            self.tag_output("avg_p", np.mean(self.bc_pressure.vector()[:]))

            # Average of all initialized fields (confirms function assignment) ### Depends on DOFS
            self.tag_output("min_initial_values", np.min(self.u0.vector()[:]))
            self.tag_output("max_initial_values", np.max(self.u0.vector()[:]))
            self.tag_output("avg_initial_values", np.mean(self.u0.vector()[:]))

            # Get number of boundary conditions 
            num_bc = len(self.bcu) + len(self.bcp) + len(self.bcs)
            self.tag_output("num_bc",num_bc)

    def SetupBoundaries(self):
        ### Create the equations need for defining the boundary conditions ###
        ### this is sloppy and will be cleaned up.
        ### Inflow is always from the front

        self.fprint("Applying Boundary Conditions",offset=1)
        unique_ids = self.dom.boundary_subdomains

        ### Assemble boundary conditions ###
        bcu_eqns = []
        bcp_eqns = []
        for bc_type, bs in self.boundary_types.items():
            if bc_type == "inflow":
                for b in bs:
                    if self.boundary_names[b] in unique_ids:
                        bcu_eqns.append([self.fs.V, self.fs.W.sub(0), self.bc_velocity, self.boundary_names[b]])

            elif bc_type == "no_slip":
                for b in bs:
                    bcu_eqns.append([self.fs.V, self.fs.W.sub(0), self.zeros, self.boundary_names[b]])

            elif bc_type == "free_slip":
                temp_list = list(self.boundary_names.keys()) # get ordered list
                for b in bs:

                    ### get a facet on the relevant boundary ###
                    boundary_id = self.boundary_names[b]

                    ### check to make sure the free slip boundary still exists ###
                    if boundary_id in unique_ids:

                        facet_ids = self.dom.boundary_markers.where_equal(boundary_id)
                        test_facet = Facet(self.dom.mesh,facet_ids[int(len(facet_ids)/2.0)])

                        ### get the function space sub form the normal ###
                        facet_normal = test_facet.normal().array()
                        field_id = int(np.argmin(abs(abs(facet_normal)-1.0)))

                        bcu_eqns.append([self.fs.V.sub(field_id), self.fs.W.sub(0).sub(field_id), self.zero, boundary_id])

            elif bc_type == "no_stress":
                for b in bs:
                    bcu_eqns.append([None, None, None, self.boundary_names[b]])
                    bcp_eqns.append([self.fs.Q, self.fs.W.sub(1), self.zero, self.boundary_names[b]])

            else:
                raise ValueError(bc_type+" is not a recognized boundary type")
        bcs_eqns = bcu_eqns#+bcp_eqns

        ### Set the boundary conditions ###
        self.bcu = []
        for i in range(len(bcu_eqns)):
            if bcu_eqns[i][0] is not None:
                self.bcu.append(DirichletBC(bcu_eqns[i][0], bcu_eqns[i][2], bcu_eqns[i][3]))

        self.bcp = []
        for i in range(len(bcp_eqns)):
            if bcp_eqns[i][0] is not None:
                self.bcp.append(DirichletBC(bcp_eqns[i][0], bcp_eqns[i][2], bcp_eqns[i][3]))

        self.bcs = []
        for i in range(len(bcs_eqns)):
            if bcs_eqns[i][0] is not None:
                self.bcs.append(DirichletBC(bcs_eqns[i][1], bcs_eqns[i][2], bcs_eqns[i][3]))

        self.fprint("Boundary Conditions Applied",offset=1)
        self.fprint("")

    def PrepareVelocity(self,inflow_angle):
        length = len(self.unit_reference_velocity)
        ux_com = np.zeros(length)
        uy_com = np.zeros(length)
        uz_com = np.zeros(length)

        for i in range(length):
            v = self.HH_vel * self.unit_reference_velocity[i]
            ux_com[i] = math.cos(inflow_angle)*v
            uy_com[i] = math.sin(inflow_angle)*v
            if self.dom.dim == 3:
                uz_com[i] = 0.0   
        return [ux_com,uy_com,uz_com]

    def RecomputeVelocity(self,inflow_angle):
        self.fprint("Recomputing Velocity")
        ux_com, uy_com, uz_com = self.PrepareVelocity(inflow_angle)

        self.ux = Function(self.fs.V0)
        self.uy = Function(self.fs.V1)
        if self.dom.dim == 3:
            self.uz = Function(self.fs.V2)

        self.ux.vector()[:] = ux_com
        self.uy.vector()[:] = uy_com
        if self.dom.dim == 3:
            self.uz.vector()[:] = uz_com

        ### Assigning Velocity
        self.bc_velocity = Function(self.fs.V)
        self.bc_velocity.rename("bc_velocity","bc_velocity")
        
        if self.dom.dim == 3:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy,self.uz])
        else:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy])
        
        ### Create Pressure Boundary Function
        self.bc_pressure = Function(self.fs.Q)

        ### Create Initial Guess
        self.fprint("Assigning Initial Guess")
        self.u0 = Function(self.fs.W)
        self.fs.SolutionAssigner.assign(self.u0,[self.bc_velocity,self.bc_pressure])

        self.SetupBoundaries()

    def UpdateVelocity(self, simTime):
        pass

    def SaveInitialGuess(self,val=0):
        """
        This function saves the turbine force if exists to output/.../functions/
        """
        self.bc_velocity.vector()[:]=self.bc_velocity.vector()[:]/self.dom.xscale
        self.dom.mesh.coordinates.dat.data[:]=self.dom.mesh.coordinates.dat.data[:]/self.dom.xscale

        if self.ig_first_save:
            self.u0_file = self.params.Save(self.bc_velocity,"u0",subfolder="functions/",val=val)
            self.p0_file = self.params.Save(self.bc_pressure,"p0",subfolder="functions/",val=val)
            self.ig_first_save = False
        else:
            self.params.Save(self.bc_velocity,"u0",subfolder="functions/",val=val,file=self.u0_file)
            self.params.Save(self.bc_pressure,"p0",subfolder="functions/",val=val,file=self.p0_file)
        self.bc_velocity.vector()[:]=self.bc_velocity.vector()[:]*self.dom.xscale
        self.dom.mesh.coordinates.dat.data[:]=self.dom.mesh.coordinates.dat.data[:]*self.dom.xscale

    def SaveHeight(self,val=0):
        """
        This function saves the turbine force if exists to output/.../functions/
        """
        self.dom.mesh.coordinates.dat.data[:]=self.dom.mesh.coordinates.dat.data[:]/self.dom.xscale
        self.height.vector()[:]=self.height.vector()[:]/self.dom.xscale
        self.depth.vector()[:]=self.depth.vector()[:]/self.dom.xscale
        if self.height_first_save:
            self.height_file = self.params.Save(self.height,"height",subfolder="functions/",val=val)
            self.depth_file = self.params.Save(self.depth,"depth",subfolder="functions/",val=val)
            self.height_first_save = False
        else:
            self.params.Save(self.height,"height",subfolder="functions/",val=val,file=self.height_file)
            self.params.Save(self.depth,"depth",subfolder="functions/",val=val,file=self.depth_file)
        self.height.vector()[:]=self.height.vector()[:]*self.dom.xscale
        self.depth.vector()[:]=self.depth.vector()[:]*self.dom.xscale
        self.dom.mesh.coordinates.dat.data[:]=self.dom.mesh.coordinates.dat.data[:]*self.dom.xscale

    def CalculateHeights(self):
        ### Calculate the distance to the ground for the Q function space ###
        # self.z_dist_Q = Function(fs.Q)
        self.height = Function(self.fs.Q)
        self.depth = Function(self.fs.Q)
        Q_coords = self.fs.Q.tabulate_dof_coordinates()
        height_vals = self.height.vector()[:]
        for i in range(len(Q_coords)):
            height_vals[i] = self.dom.Ground(Q_coords[i,0],Q_coords[i,1])
        z_dist_Q = Q_coords[:,2]-height_vals
        self.height.vector()[:]=height_vals
        self.depth.vector()[:]=z_dist_Q

        ### Calculate the distance to the ground for the V function space ###
        self.depth_V = Function(self.fs.V)
        V_coords = self.fs.V.tabulate_dof_coordinates()
        z_dist_V_val = np.zeros(len(V_coords))
        for i in range(len(V_coords)):
            z_dist_V_val[i] = V_coords[i,2]-self.dom.Ground(V_coords[i,0],V_coords[i,1])
        self.depth_V.vector()[:]=z_dist_V_val

        self.V0_coords = self.fs.V0.tabulate_dof_coordinates()


class UniformInflow(GenericBoundary):
    def __init__(self,dom,fs,farm):
        super(UniformInflow, self).__init__(dom,fs,farm)
        self.fprint("Setting Up Boundary Conditions",special="header")
        self.fprint("Type: Uniform Inflow")
        for key, values in self.boundary_types.items():
            self.fprint("Boundary Type: {0}, Applied to:".format(key))
            for value in values:
                self.fprint(value,offset=1)
        ### Create the Velocity Function ###
        self.ux = Function(fs.V0)
        self.uy = Function(fs.V1)
        if self.dom.dim == 3:
            self.uz = Function(fs.V2)
        self.unit_reference_velocity = np.full(len(self.ux.vector()[:]),1.0)
        self.ux.vector()[:] = self.unit_reference_velocity

        ux_com, uy_com, uz_com = self.PrepareVelocity(self.dom.inflow_angle)
        self.ux.vector()[:] = ux_com
        self.uy.vector()[:] = uy_com
        if self.dom.dim == 3:
            self.uz.vector()[:] = uz_com

        ### Compute distances ###
        if self.dom.dim == 3:
            self.fprint("Computing Distance to Ground")
            self.CalculateHeights()

        ### Assigning Velocity
        self.fprint("Computing Velocity Vector")
        self.bc_velocity = Function(fs.V)
        temp = np.zeros(self.bc_velocity.vector().get_local().shape)
        if self.dom.dim == 3:
            temp[0::3] = self.ux.vector().get_local()
            temp[1::3] = self.uy.vector().get_local()
            temp[2::3] = self.uz.vector().get_local()
        else:
            temp[0::2] = self.ux.vector().get_local()
            temp[1::2] = self.uy.vector().get_local()
        self.bc_velocity.vector().set_local(temp)

        ### Create Pressure Boundary Function
        self.bc_pressure = Function(fs.Q)

        ### Create Initial Guess
        self.fprint("Assigning Initial Guess")
        self.u0 = Function(fs.W)
        self.u0.sub(0).vector().set_local(self.bc_velocity.vector().get_local())
        self.u0.sub(1).vector().set_local(self.bc_pressure.vector().get_local())

        ### Setup the boundary Conditions ###
        self.SetupBoundaries()
        self.DebugOutput()
        self.fprint("Boundary Condition Finished",special="footer")

class PowerInflow(GenericBoundary):
    """
    PowerInflow creates a set of boundary conditions where the x-component
    of velocity follows a power law. Currently the function is 

    .. math::

        u_x=8.0 \\left( \\frac{z-z_0}{z_1-z_0} \\right)^{0.15}.
        
    where :math:`z_0` is the ground and :math:`z_1` is the top of the domain.

    Args:
        dom (:class:`windse.DomainManager.GenericDomain`): A windse domain object.
        fs (:class:`windse.FunctionSpaceManager.GenericFunctionSpace`): 
            A windse function space object

    Todo:
        * Make the max velocity an input
        * Make the power an input
    """
    def __init__(self,dom,fs,farm):
        super(PowerInflow, self).__init__(dom,fs,farm)

        if self.dom.dim != 3:
            raise ValueError("PowerInflow can only be used with 3D domains.")

        ### Setup Boundary Conditions
        self.fprint("Setting Up Boundary Conditions",special="header")
        self.fprint("Type: Power Law Inflow")
        for key, values in self.boundary_types.items():
            self.fprint("Boundary Type: {0}, Applied to:".format(key))
            for value in values:
                self.fprint(value,offset=1)
        self.fprint("")

        ### Compute distances ###
        self.fprint("Computing Distance to Ground")
        self.CalculateHeights()
        depth_v0,depth_v1,depth_v2 = self.depth_V.split(deepcopy=True)

        ### Create the Velocity Function ###
        self.fprint("Computing Velocity Vector")
        self.ux = Function(fs.V0)
        self.uy = Function(fs.V1)
        self.uz = Function(fs.V2)
        
        #################
        #################
        #################
        #################
        #################
        #################
        scaled_depth = np.abs(np.divide(depth_v0.vector()[:],(np.mean(self.vel_height)-dom.ground_reference)))
        # scaled_depth = np.abs(np.divide(depth_v0.vector()[:],(np.mean(self.vel_height)-0.0)))
        #################
        #################
        #################
        #################
        #################

        self.unit_reference_velocity = np.power(scaled_depth,self.power)
        # self.reference_velocity = np.multiply(self.HH_vel,np.power(scaled_depth,self.power))
        ux_com, uy_com, uz_com = self.PrepareVelocity(self.dom.inflow_angle)

        self.ux.vector()[:] = ux_com
        self.uy.vector()[:] = uy_com
        self.uz.vector()[:] = uz_com

        ### Assigning Velocity
        self.bc_velocity = Function(self.fs.V)
        if self.dom.dim == 3:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy,self.uz])
        else:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy])

        ### Create Pressure Boundary Function
        self.bc_pressure = Function(self.fs.Q)

        ### Create Initial Guess
        self.fprint("Assigning Initial Guess")
        self.u0 = Function(self.fs.W)
        self.fs.SolutionAssigner.assign(self.u0,[self.bc_velocity,self.bc_pressure])

        ### Setup the boundary Conditions ###
        self.SetupBoundaries()
        self.DebugOutput()
        self.fprint("Boundary Condition Setup",special="footer")

class LogLayerInflow(GenericBoundary):
    def __init__(self,dom,fs,farm):
        super(LogLayerInflow, self).__init__(dom,fs,farm)

        if self.dom.dim != 3:
            raise ValueError("LogLayerInflow can only be used with 3D domains.")

        ### Setup Boundary Conditions
        self.fprint("Setting Up Boundary Conditions",special="header")
        self.fprint("Type: Power Law Inflow")

        for key, values in self.boundary_types.items():
            self.fprint("Boundary Type: {0}, Applied to:".format(key))
            for value in values:
                self.fprint(value,offset=1)
        self.fprint("")

        ### Compute distances ###
        self.fprint("Computing Distance to Ground")
        self.CalculateHeights()
        depth_v0,depth_v1,depth_v2 = self.depth_V.split(deepcopy=True)



        ### Create the Velocity Function ###
        self.fprint("Computing Velocity Vector")
        self.ux = Function(fs.V0)
        self.uy = Function(fs.V1)
        self.uz = Function(fs.V2)
        if dom.ground_reference == 0:
            scaled_depth = np.abs(np.divide(depth_v0.vector()[:]+0.0001,0.0001))
            ustar = self.k/np.log(np.mean(self.vel_height)/0.0001)
        elif dom.ground_reference <= 0:
            raise ValueError("Log profile cannot be used with negative z values")
        else:
            scaled_depth = np.abs(np.divide(depth_v0.vector()[:]+dom.ground_reference,(dom.ground_reference)))
            ustar = self.k/np.log(np.mean(self.vel_height)/dom.ground_reference)
        self.unit_reference_velocity = np.multiply(ustar/self.k,np.log(scaled_depth))
        ux_com, uy_com, uz_com = self.PrepareVelocity(self.dom.inflow_angle)

        self.ux.vector()[:] = ux_com
        self.uy.vector()[:] = uy_com
        self.uz.vector()[:] = uz_com

        ### Assigning Velocity
        self.bc_velocity = Function(self.fs.V)
        if self.dom.dim == 3:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy,self.uz])
        else:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy])

        ### Create Pressure Boundary Function
        self.bc_pressure = Function(self.fs.Q)

        ### Create Initial Guess
        self.fprint("Assigning Initial Guess")
        self.u0 = Function(self.fs.W)
        self.fs.SolutionAssigner.assign(self.u0,[self.bc_velocity,self.bc_pressure])

        ### Setup the boundary Conditions ###
        self.SetupBoundaries()
        self.DebugOutput()
        self.fprint("Boundary Condition Setup",special="footer")

class TurbSimInflow(LogLayerInflow):
    def __init__(self,dom,fs,farm):
        super(TurbSimInflow, self).__init__(dom,fs,farm)

        ### Get the path for turbsim data ###
        if self.turbsim_path is None:
            raise ValueError("Please provide the path to the turbsim data")

        ### Load Turbsim Data ###
        uTotal = np.load(self.turbsim_path+'turb_u.npy')
        vTotal = np.load(self.turbsim_path+'turb_v.npy')
        wTotal = np.load(self.turbsim_path+'turb_w.npy')

        ### Extract number of data points ###
        ny = np.shape(uTotal)[1]
        nz = np.shape(uTotal)[0]
        nt = np.shape(uTotal)[2]

        ### Create the data bounds ###
        y = np.linspace(self.dom.y_range[0], self.dom.y_range[1], ny)
        z = np.linspace(self.dom.z_range[0], self.dom.z_range[1], nz)
        t = np.linspace(0.0, self.final_time, nt)

        ### Build interpolating functions ###
        self.interp_u = RegularGridInterpolator((z, y, t), uTotal)
        self.interp_v = RegularGridInterpolator((z, y, t), vTotal)
        self.interp_w = RegularGridInterpolator((z, y, t), wTotal)

        ### Locate Boundary DOFS indexes ###
        # Define tolerance
        tol = 1e-6

        ##### FIX MAKE WORK FOR ALL BOUNDARY INFLOW ####
        # Iterate and fine the boundary IDs
        self.boundaryIDs = []
        for k, pos in enumerate(self.V0_coords):
            if pos[0] < self.dom.x_range[0] + tol:
                self.boundaryIDs.append(k)

        self.UpdateVelocity(0.0)
        self.DebugOutput()

    def UpdateVelocity(self, simTime):

        # Define tolerance
        tol = 1e-6

        # Interpolate a value at each boundary coordinate
        for k in self.boundaryIDs:
            # Get the position corresponding to this boundary id
            pos = self.V0_coords[k, :]

            # The interpolation point specifies a 3D (z, y, time) point
            xi = np.array([pos[2], pos[1], simTime])

            # Get the interpolated value at this point
            self.ux.vector()[k] = self.interp_u(xi)
            self.uy.vector()[k] = self.interp_v(xi)
            self.uz.vector()[k] = self.interp_w(xi)

        ### Assigning Velocity
        self.bc_velocity = Function(self.fs.V)
        if self.dom.dim == 3:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy,self.uz])
        else:
            self.fs.VelocityAssigner.assign(self.bc_velocity,[self.ux,self.uy])

        self.SetupBoundaries()













