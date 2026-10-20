#---Preamble----#
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


#Experiment concept
#For a range of H_x and J values, apply J as pulses with a specific pulsewidth and allow the system to relax for some time after the pulse
#Do not reinitialise the system after each pulse
pw = 1e-9
rt = 2e-9
Exp_num = 1

while os.path.exists(f"Experiment_{Exp_num}"):
	Exp_num+=1
os.mkdir(f"Experiment_{Exp_num}")

H_list = np.linspace(20,40,20)
#H_list = [-30,30]
J_list_pos = np.linspace(-3e12,-4e12,20)
J_list_neg = J_list_pos[::-1]+1 
#J_list = np.concatenate((J_list_pos,J_list_neg))
J_list = J_list_pos
for h in H_list:
	folder_name = f"Experiment_{Exp_num}\\Hx_{h}"
	os.mkdir(folder_name)
	M_z = []
	H = (h*(1e-3)/mm.consts.mu0,0,0)
	region = df.Region(p1=p1,p2=p2)
	mesh = df.Mesh(region = region, cell = cell)
	system = mm.System(name = "CIMR_pulsed")
	system.energy = mm.Exchange(A=A)+mm.DMI(D = D,crystalclass = "Cnv_z")+mm.UniaxialAnisotropy(K=K,u=u)
	system.m = df.Field(mesh,nvdim =3,value = (0,0,1),norm = M_s)

	system.m.z.sel("z").mpl(scalar_kw={"vmin": -M_s, "vmax": M_s})
	plt.savefig(f"{folder_name}\\initial_config.png")
	plt.close()
	md = oc.MinDriver()
	md.drive(system)

	system.m.z.sel("z").mpl(scalar_kw={"vmin": -M_s, "vmax": M_s})
	plt.savefig(f"{folder_name}\\post_energymin.png")
	plt.close()
	system.energy +=mm.Zeeman(H=H)
	for j in J_list:

		os.mkdir(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}")
		system.dynamics = mm.Precession(gamma0 = gamma0)+mm.Damping(alpha = alpha)+mm.Slonczewski(J = j,mp = mp,P = P,Lambda = 1)
		td = oc.TimeDriver()
		td.drive(system,t=pw,n=10)
		data = mdata.Data("CIMR_pulsed")[-1]

		for d in range(1):
			data[d].z.sel("z").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\pulse_zshot_{d}")
			#data[d].y.sel("y").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			#plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\pulse_yshot_{d}")
			#data[d].x.sel("x").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			#plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\pulse_xshot_{d}")

			plt.close()

		system.dynamics = mm.Precession(gamma0 = gamma0)+mm.Damping(alpha = alpha)
		td.drive(system,t=rt,n=1)
		for d in range(1):
			data[d].z.sel("z").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\relax_zshot_{d}")
			#data[d].y.sel("y").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			#plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\relax_yshot_{d}")
			#data[d].x.sel("x").mpl(scalar_kw = {"vmin":-M_s,"vmax":M_s})
			#plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\relax_xshot_{d}")
			plt.close()

		M_z.append(system.m.orientation.mean()[2])
		params = ["Simulation parameters\n","-----------------",f"film dimensions(L,W,T):{L},{w},{d}\n",f"Ms = {M_s}\n",f"A = {A}\n",f"D = {D}\n",f"K = {K}\n",f"gamma0 = {gamma0}\n",f"alpha = {alpha}\n",f"P = {P}\n",f"J = {j}\n",f"H_x = {H[0]}\n",f"time = {t}s\n","comment :"]
		with open(f"Experiment_{Exp_num}\\Hx_{h}\\J_{j/1e12}\\parameters.txt",'w') as f:
			f.writelines(params)
	
	with open(f"Experiment_{Exp_num}\\MH_data_Hx_{h}.txt",'w') as f:
		for q in range(len(J_list)):
			f.writelines(f"{J_list[q]},{M_z[q]}\n")

	plt.plot(J_list,M_z)
	plt.savefig(f"Experiment_{Exp_num}\\Hx_{h}\\MH_plot.png")
	plt.close()