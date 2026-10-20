import numpy as np
import matplotlib.pyplot as plt
import oommfc as oc
import discretisedfield as df
import micromagneticmodel as mm
import micromagneticdata as mdata
import os

L = 100e-9 #sample length (m)
w = 10e-9 #sample width (m)
d = 0.59e-9 #discretisation cell size (m)
M_s = 1.156e6
A = 16e-12
D = 1.1e-3
K = 0.25e6
u = (0,0,1)
#Hx_mT = 50
#H = (Hx_mT*1e-3/mm.consts.mu0, 0, 0)   # ≈ 3.98e4 A/m for 50 mT
#H = (1000,0,100)
gamma0 = 2.211e5
alpha = 0.3 
P = 0.09
mp = (0,1,0)
#J = 4.08e12 #current density A/m^2
t = 0.5e-9
#beta = np.abs(mm.consts.h/(mm.consts.mu0*mm.consts.e))*(J/(d*M_s))
#Mesh def
p1 = (0,0,0)
p2 = (L,w,d)
cell = (2e-9,2e-9,d)
#-----Experiment Concept------
#Vary the J and find J_c for a particular value of H_x
#First I give a loop for varying H_x and then a loop for varying J
#Store images for each at short time intervals
#J step size 4e10
#H_x step size 10 mT

Exp_num = 2
J_list_pos = np.linspace(5e12,-5e12,10)
J_list_neg = J_list_pos[::-1]+1 
J_list = np.concatenate((J_list_pos,J_list_neg))
H_list = [-30,30]

if os.path.exists(f"Experiment_{Exp_num}"):
	Exp_num += 1
	os.mkdir(f"Experiment_{Exp_num}")
else:
	os.mkdir(f"Experiment_{Exp_num}")

for i in H_list:
	Hx_mT = i
	H = (Hx_mT*1e-3/mm.consts.mu0, 0, 0)
	M_z = []
	for j in J_list:
		J = j
		folder_name = f"Experiment_{Exp_num}\\Hx_{H}_J_{j}"
		os.mkdir(folder_name)

		#Defining the system
		
		region = df.Region(p1=p1,p2=p2)
		mesh = df.Mesh(region = region, cell = cell)
		system = mm.System(name = "CIMR")
		system.energy = mm.Exchange(A=A)+mm.DMI(D = D,crystalclass = "Cnv_z")+mm.UniaxialAnisotropy(K=K,u=u)
		system.dynamics = mm.Precession(gamma0 = gamma0)+mm.Damping(alpha = alpha)+mm.Slonczewski(J = J,mp = mp,P = P,Lambda = 1)
		system.m = df.Field(mesh,nvdim =3,value = (0,0,-1),norm = M_s)

		system.m.z.sel("z").mpl(scalar_kw={"vmin": -M_s, "vmax": M_s})
		plt.savefig(f"{folder_name}\\initial_config.png")
		plt.close()
		#Energy minimisation
		md = oc.MinDriver()
		md.drive(system)


		system.m.z.sel("z").mpl(scalar_kw={"vmin": -M_s, "vmax": M_s})
		plt.savefig(f"{folder_name}\\post_energymin.png")
		plt.close()

		system.energy += mm.Zeeman(H=H)
		#Time evolution

		td = oc.TimeDriver()
		td.drive(system, t=t,n=10)

		system.dynamics = mm.Precession(gamma0 = gamma0)+mm.Damping(alpha = alpha)
		data = mdata.Data("CIMR")[-1]
		for l in range(10):
			data[l].z.sel("z").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			plt.savefig(f"{folder_name}\\shot_{l}.png")
			plt.close()
		td.drive(system, t=t,n=10)
		data = mdata.Data("CIMR")[-1]
		for l in range(10):
			data[l].z.sel("z").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			plt.savefig(f"{folder_name}\\postpulse_shot_{l}.png")
			plt.close()

		M_z.append(system.m.orientation.mean()[2])

		params = ["Simulation parameters\n","-----------------",f"film dimensions(L,W,T):{L},{w},{d}\n",f"Ms = {M_s}\n",f"A = {A}\n",f"D = {D}\n",f"K = {K}\n",f"gamma0 = {gamma0}\n",f"alpha = {alpha}\n",f"P = {P}\n",f"J = {J}\n",f"H_x = {H[0]}\n",f"time = {t}s\n","comment : First doing energy minimisation for 0 Hx field with +z saturation then adding zeeman term after the minimisation."]
		with open(f"{folder_name}\\parameters.txt",'w') as f:
			f.writelines(params)
	plt.plot(J_list,M_z)
	plt.savefig(f"Experiment_{Exp_num}\\MvJ_Hx_{H[0]}.png")
	plt.close()


#for k in range(50):
#	last_iter[k].z.sel("z").mpl(scalar_kw={"vmin": -M_s, "vmax": M_s})
#	plt.savefig(f"{folder_name}\\shot_{i}.png")
#	plt.close()