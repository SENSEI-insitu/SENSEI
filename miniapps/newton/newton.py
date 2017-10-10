from mpi4py import *
import sensei
import vtk, vtk.util.numpy_support as vtknp
import numpy as np, sys, os, argparse
np.seterr(divide='ignore', invalid='ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

class initial_condition:
    def __init__(self):
        self.n_bodies = 0
        self.time_step = 0
        self.id = 0

    def get_number_of_bodies(self):
        return self.n_bodies

    def set_number_of_bodies(self, n):
        self.n_bodies = n

    def set_particle_id(self, i):
        self.id = i

    def get_particle_id(self):
        return self.id

    def get_time_step(self):
        return self.time_step

    def set_time_step(self, dt):
        self.time_step = dt

    def intialize_bodies(self, x,y,z,m,vx,vy,vz):
        return

    def allocate(self):
        n = self.get_number_of_bodies()
        ids = np.arange(self.id,self.id+n)
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        m = np.zeros(n)
        vx = np.zeros(n)
        vy = np.zeros(n)
        vz = np.zeros(n)
        fx = np.zeros(n)
        fy = np.zeros(n)
        fz = np.zeros(n)
        self.initialize_bodies(x,y,z,m,vx,vy,vz)
        F(x,y,z,m,fx,fy,fz)
        return ids,x,y,z,m,vx,vy,vz,fx,fy,fz

class uniform_random_ic(initial_condition):
    def __init__(self, npts, x0,x1,y0,y1,m0,m1,v0,v1):
        self.n_bodies_global = npts
        n_lg = npts % n_ranks
        n_sm = n_ranks - n_lg
        npts_sm = npts / n_ranks
        self.n_bodies_local = npts_sm + (1 if rank > n_sm else 0)
        self.set_number_of_bodies(self.n_bodies_local)
        self.set_time_step(4*24*3600)
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.m0 = m0
        self.m1 = m1
        self.v0 = v0
        self.v1 = v1
        self.set_particle_id(rank * npts_sm + \
            (rank - n_sm if rank > npts_sm else 0))

    def __str__(self):
        return 'uniform_random_ic'

    def initialize_bodies(self, x,y,z,m,vx,vy,vz):
        npa = self.get_number_of_bodies()
        if npa < 1:
            return
        x[:] = (self.x1 - self.x0) * np.random.random_sample(npa) + self.x0
        y[:] = (self.y1 - self.y0) * np.random.random_sample(npa) + self.y0
        m[:] = (self.m1 - self.m0) * np.random.random_sample(npa) + self.m0
        r = np.sqrt(x**2 + y**2)
        v0 = (self.v1 - self.v0) * np.random.random_sample(npa) + self.v0
        vx[:] = v0*(-y - x*0.1)/r
        vy[:] = v0*(x - y*0.1)/r
        # put sun at the origin
        if rank == 0:
            x[0] = 0
            y[0] = 0
            vx[0] = 0
            vy[0] = 0
            m[0] = 1.989e30

# Velocity Verlet
# v_{n+1/2} = v_n + (h/2)*F(x_n)
# x_{n+1} = x_n + h*v_{n+1/2}
# v_{n+1} = v_{n+1/2} + (h/2)*F(x_{n+1})
def velocity_verlet(x,y,z,m,vx,vy,vz,fx,fy,fz,h):
    # half step in velocity
    h2 = 0.5*h
    vx += h2*fx/m
    vy += h2*fy/m
    vz += h2*fz/m
    # step position
    x += h*vx
    y += h*vy
    z += h*vz
    # update forces
    F(x,y,z,m,fx,fy,fz)
    # finish velocity step
    vx += h2*fx/m
    vy += h2*fy/m
    vz += h2*fz/m
    return

def F(x,y,z,m,fx,fy,fz):
    fx.fill(0)
    fy.fill(0)
    fz.fill(0)
    j = 0
    while j < n_ranks:
        i = j
        while i < n_ranks:
            accumulate_forces(i,j,x,y,z,m,fx,fy,fz)
            i += 1
        j += 1
    return

def accumulate_forces(src_rank, dest_rank, x, y, z, m, fx, fy, fz):
    if (rank == src_rank) and (rank == dest_rank):
        # calc force
        fxji = np.zeros_like(x)
        fyji = np.zeros_like(x)
        fzji = np.zeros_like(x)
        calc_forces(x,y,z,m,x,y,z,m,fxji,fyji,fzji)
        # accumulate
        fx += fxji
        fy += fyji
        fz += fzji
        return
    elif rank == src_rank:
        # send position and mass
        ni = int(len(x))
        comm.send(ni, dest=dest_rank, tag=3330)
        comm.Send(x, dest_rank, 3331)
        comm.Send(y, dest_rank, 3332)
        comm.Send(z, dest_rank, 3333)
        comm.Send(m, dest_rank, 3334)
        # receive forces from remote bodies
        fxji = np.zeros_like(x)
        fyji = np.zeros_like(x)
        fzji = np.zeros_like(x)
        comm.Recv(fxji, dest_rank, 3335)
        comm.Recv(fyji, dest_rank, 3336)
        comm.Recv(fzji, dest_rank, 3337)
        # accumulate
        fx += fxji
        fy += fyji
        fz += fzji
        return
    elif rank == dest_rank:
        # receive position and mass
        ni = comm.recv(source=src_rank, tag=3330)
        xi = np.zeros(ni, x.dtype)
        yi = np.zeros(ni, x.dtype)
        zi = np.zeros(ni, x.dtype)
        mi = np.zeros(ni, x.dtype)
        comm.Recv(xi, src_rank, 3331)
        comm.Recv(yi, src_rank, 3332)
        comm.Recv(zi, src_rank, 3333)
        comm.Recv(mi, src_rank, 3334)
        # calc forces from us on them
        fxji = np.zeros(ni, x.dtype)
        fyji = np.zeros(ni, x.dtype)
        fzji = np.zeros(ni, x.dtype)
        calc_forces(xi,yi,zi,mi,x,y,z,m,fxji,fyji,fzji)
        # send forces back
        comm.Send(fxji, src_rank, 3335)
        comm.Send(fyji, src_rank, 3336)
        comm.Send(fzji, src_rank, 3337)
        # calc forces from them on us
        fxji = np.zeros_like(x)
        fyji = np.zeros_like(x)
        fzji = np.zeros_like(x)
        calc_forces(x,y,z,m,xi,yi,zi,mi,fxji,fyji,fzji)
        # accumulate
        fx += fxji
        fy += fyji
        fz += fzji
        return
    return

# returns force on each xi by all xj
def calc_forces(xi,yi,zi,mi, xj,yj,zj,mj, fxji, fyji, fzji):
    ni = len(xi)
    q = 0
    while q < ni:
        # the force applied by objects j on object i
        # F_i = sum_j(-n((G*m_i*m_j)/mag(r_i,r_j)**2)*unit(r_i, r_j))
        dx = xj - xi[q]
        dy = yj - yi[q]
        dz = zj - zi[q]
        mag2 = dx**2 + dy**2 + dz**2
        mag = np.sqrt(mag2)
        not_same = np.logical_not(np.isclose(mag, 0.0))
        ux = np.where(not_same, dx/mag, 0)
        uy = np.where(not_same, dy/mag, 0)
        uz = np.where(not_same, dz/mag, 0)
        G = 6.67408e-11
        G_mij_mag2 = np.where(not_same, G*mi[q]*mj/mag2, 0)
        fxji[q] = np.sum(ux*G_mij_mag2)
        fyji[q] = np.sum(uy*G_mij_mag2)
        fzji[q] = np.sum(uz*G_mij_mag2)
        q += 1
    return

def points_to_polydata(ids,x,y,z,m,vx,vy,vz,fx,fy,fz):
    nx = len(x)
    # points
    xyz = np.zeros(3*nx, dtype=np.float32)
    xyz[::3] = x[:]
    xyz[1::3] = y[:]
    xyz[2::3] = z[:]
    vxyz = vtknp.numpy_to_vtk(xyz, deep=1)
    vxyz.SetNumberOfComponents(3)
    vxyz.SetNumberOfTuples(nx)
    pts = vtk.vtkPoints()
    pts.SetData(vxyz)
    # cells
    cids = np.empty(2*nx, dtype=np.int32)
    cids[::2] = 1
    cids[1::2] = np.arange(0,nx,dtype=np.int32)
    cells = vtk.vtkCellArray()
    cells.SetCells(nx, vtknp.numpy_to_vtk(cids, \
        deep=1, array_type=vtk.VTK_ID_TYPE))
    # scalars, id
    vtkids = vtknp.numpy_to_vtk(ids, 1, vtk.VTK_LONG)
    vtkids.SetName('ids')
    # mass
    vtkm = vtknp.numpy_to_vtk(m, deep=1)
    vtkm.SetName('m')
    # velocity
    vxyz = np.zeros(3*nx, dtype=np.float32)
    vxyz[::3] = vx
    vxyz[1::3] = vy
    vxyz[2::3] = vz
    vtkv = vtknp.numpy_to_vtk(vxyz, deep=1)
    vtkv.SetName('v')
    # mag velocity
    mv = np.sqrt(vx**2 + vy**2 + vz**2)
    vtkmv = vtknp.numpy_to_vtk(mv, deep=1)
    vtkmv.SetName('magv')
    # force
    fxyz = np.zeros(3*nx, dtype=np.float32)
    fxyz[::3] = fx
    fxyz[1::3] = fy
    fxyz[2::3] = fz
    vtkf = vtknp.numpy_to_vtk(fxyz, deep=1)
    vtkf.SetName('f')
    # mag force
    mv = np.sqrt(fx**2 + fy**2 + fz**2)
    vtkmf = vtknp.numpy_to_vtk(mv, deep=1)
    vtkmf.SetName('magf')
    # package it all up in a poly data set
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.GetPointData().AddArray(vtkids)
    pd.GetPointData().AddArray(vtkm)
    pd.GetPointData().AddArray(vtkv)
    pd.GetPointData().AddArray(vtkmv)
    pd.GetPointData().AddArray(vtkf)
    pd.GetPointData().AddArray(vtkmf)
    pd.SetVerts(cells)
    return pd

def csv_str_to_dict(s):
    d = {}
    nvp=s.split(',')
    for nv in nvp:
        nv=nv.split('=')
        if len(nv) > 1:
            d[nv[0]] = nv[1]
        else:
            d[nv[0]] = None
    return d

def check_arg(dic, arg, dfl=None, req=True):
    if not arg in dic:
        if req and dfl is None:
            status('ERROR: %s is a required argument\n'%(arg))
            return False
        else:
            dic[arg] = dfl
            return True
    return True

class analysis_adaptor:
    def __init__(self):
        self.DataAdaptor = sensei.VTKDataAdaptor.New()
        self.AnalysisAdaptor = None

    def initialize(self, analysis, args=''):
        self.Analysis = analysis
	args = csv_str_to_dict(args)
        # Libsim
        if analysis == 'libsim':
            imProps = sensei.LibsimImageProperties()
            self.AnalysisAdaptor = sensei.LibsimAnalysisAdaptor.New()
            self.AnalysisAdaptor.AddPlots('Pseudocolor','ids', False,False, \
		(0.,0.,0.),(1.,1.,1.),imProps)
        # Catalyst
        if analysis == 'catalyst':
            if check_arg(args,'script'):
                self.AnalysisAdaptor = sensei.CatalystAnalysisAdaptor.New()
                self.AnalysisAdaptor.AddPythonScriptPipeline(args['script'])
        # VTK I/O
        elif analysis == 'posthoc':
            if check_arg(args,'file','newton') and check_arg(args,'dir','./') \
                and check_arg(args,'mode','0') and check_arg(args,'freq','1'):
                self.AnalysisAdaptor = sensei.VTKPosthocIO.New()
                self.AnalysisAdaptor.Initialize(comm, args['dir'],args['file'],\
                    [], ['ids','fx','fy','fz','f','vx','vy','vz','v','m'], \
                    int(args['mode']), int(args['freq']))
        # Libisim, ADIOS, etc
        elif analysis == 'configurable':
            if check_arg(args,'config'):
                self.AnalysisAdaptor = sensei.ConfigurableAnalysis.New()
                self.AnalysisAdaptor.Initialize(comm, args['config'])

        if self.AnalysisAdaptor is None:
            status('ERROR: Failed to initialize "%s"\n'%(analysis))
            sys.exit(-1)

    def finalize(self):
        if self.Analysis == 'posthoc':
            self.AnalysisAdaptor.Finalize()

    def update(self, i,t,ids,x,y,z,m,vx,vy,vz,fx,fy,fz):

        status('% 5d\n'%(i)) if i > 0 and i % 70 == 0 else None
        status('.')

        node = points_to_polydata(ids,x,y,z,m,vx,vy,vz,fx,fy,fz)

        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(n_ranks)
        mb.SetBlock(rank, node)

        self.DataAdaptor.SetDataTime(t)
        self.DataAdaptor.SetDataTimeStep(i)
        self.DataAdaptor.SetDataObject(mb)

        self.AnalysisAdaptor.Execute(self.DataAdaptor)

        self.DataAdaptor.ReleaseData()

def status(msg):
    sys.stderr.write(msg if rank == 0 else '')

if __name__ == '__main__':
    # parse the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--analysis', type=str, default='catalyst',
        help='Type of analysis to run. posthoc,catalyst,configurable')

    parser.add_argument('--analysis_opts', type=str, default='',
        help='A CSV list of name=value pairs specific to each analysis type. ' \
             'cataylst: script=a catalyst Python script. posthoc: ' \
             'mode=0:pvd|1:visit,file=file name,dir= output dir ' \
             'freq=number of steps between I/O. configurable: config=xml file')

    parser.add_argument('--n_bodies', type=int,
        default=150, help="Number of bodies per process")

    parser.add_argument('--n_its', type=int,
        default=140, help="Number of iterations to run")

    parser.add_argument('--dt', type=float,
        help="Time step in seconds")

    args = parser.parse_args()

    # set up the initial condition
    n_bodies = args.n_bodies*n_ranks
    ic = uniform_random_ic(n_bodies, -5906.4e9, \
        5906.4e9, -5906.4e9, 5906.4e9, 10.0e24, \
        100.0e24, 1.0e3, 10.0e3)

    ids,x,y,z,m,vx,vy,vz,fx,fy,fz = ic.allocate()
    h = args.dt if args.dt else ic.get_time_step()

    # create an analysis adaptor
    adaptor = analysis_adaptor()
    adaptor.initialize(args.analysis, args.analysis_opts)

    # print the config
    status('Initialized %s. %d bodies. %d MPI ranks. ~%d ' \
        'per rank.\n%d iterations. time step %g. %s analysis.\n'%( \
        str(ic), n_bodies, n_ranks, ic.get_number_of_bodies(), \
        args.n_its, h, args.analysis))

    # run the sim and analysis
    adaptor.update(0,0,ids,x,y,z,m,vx,vy,vz,fx,fy,fz)
    i = 1
    while i <= args.n_its:
        velocity_verlet(x,y,z,m,vx,vy,vz,fx,fy,fz,h)
        adaptor.update(i,i*h,ids,x,y,z,m,vx,vy,vz,fx,fy,fz)
        i += 1

    # finish up
    adaptor.finalize()
    status('run complete\n')
