# %%
import numpy as np
import matplotlib.pyplot as plt
import random

# Initialization of parameters.
DELTA_T = 0.0001  # Unit : s
R_X = 10  # Unit : Hz
N_E = 10000
N_I = 10000
N_O = 10000
K_CNC = 1000
TIME_SPAN = 1  # Unit : s
V_THRESHOLD = 1  
TAU = 0.02  # Unit : s
BINS = int(TIME_SPAN/DELTA_T)
J_EE = 1
J_EI = -2.5
J_EX = 2
J_II = -2
J_IE = 1
J_IX = 1


# Initialization of populations of neurons 
t = np.arange(0, TIME_SPAN, DELTA_T)

V_e = np.zeros((N_E, len(t)))
spike_e = np.zeros((N_E, len(t)))

V_i = np.zeros((N_I, len(t)))
spike_i = np.zeros((N_I, len(t)))

V_x = np.random.random_sample((N_I,BINS))
spike_x = np.copy(V_x)
spike_x[spike_x > R_X * DELTA_T] = 0
spike_x[(spike_x < R_X * DELTA_T) & (spike_x != 0)] = 1/DELTA_T



# Initialization of connections between neurons and inputs to each neuron
c_ex = np.random.random_sample((N_E, N_O))
c_ex[c_ex > 1 - K_CNC/N_E] = 1
c_ex[c_ex < 1 - K_CNC/N_E] = 0
c_ee = np.random.random_sample((N_E, N_E))
c_ee[c_ee > 1 - K_CNC/N_E] = 1
c_ee[c_ee < 1 - K_CNC/N_E] = 0
c_ei = np.random.random_sample((N_E, N_I))
c_ei[c_ei > 1 - K_CNC/N_E] = 1
c_ei[c_ei < 1 - K_CNC/N_E] = 0

c_ix = np.random.random_sample((N_I, N_O))
c_ix[c_ix > 1 - K_CNC/N_I] = 1
c_ix[c_ix < 1 - K_CNC/N_I] = 0
c_ie = np.random.random_sample((N_I, N_E))
c_ie[c_ie > 1 - K_CNC/N_I] = 1
c_ie[c_ie < 1 - K_CNC/N_I] = 0
c_ii = np.random.random_sample((N_I, N_I))
c_ii[c_ii > 1 - K_CNC/N_I] = 1
c_ii[c_ii < 1 - K_CNC/N_I] = 0

h_ee = np.zeros((N_E, len(t)))
h_ei = np.zeros((N_E, len(t)))
h_ex = np.zeros((N_E, len(t)))
h_ii = np.zeros((N_I, len(t)))
h_ie = np.zeros((N_I, len(t)))
h_ix = np.zeros((N_I, len(t)))

# Integration
for j in range(1, len(t)):
    h_ee[:, j-1] = J_EE / np.sqrt(K_CNC) * np.dot(c_ee, spike_e[:, j-1])
    h_ei[:, j-1] = J_EI / np.sqrt(K_CNC) * np.dot(c_ei, spike_i[:, j-1])
    h_ex[:, j-1] = J_EX / np.sqrt(K_CNC) * np.dot(c_ex, spike_x[:, j-1])
    h_ii[:, j-1] = J_II / np.sqrt(K_CNC) * np.dot(c_ii, spike_i[:, j-1])
    h_ie[:, j-1] = J_IE / np.sqrt(K_CNC) * np.dot(c_ie, spike_e[:, j-1])
    h_ix[:, j-1] = J_IX / np.sqrt(K_CNC) * np.dot(c_ix, spike_x[:, j-1])
    V_e[:, j] = V_e[:, j-1] + DELTA_T * (-V_e[:, j-1]/TAU + h_ee[:, j-1] + h_ei[:, j-1] + h_ex[:, j-1])
    V_i[:, j] = V_i[:, j-1] + DELTA_T * (-V_i[:, j-1]/TAU + h_ii[:, j-1] + h_ie[:, j-1] + h_ix[:, j-1])

    spike_e[V_e[:, j] > V_THRESHOLD, j] = 1/DELTA_T
    V_e[V_e[:, j] > V_THRESHOLD, j] = 0
    spike_i[V_i[:, j] > V_THRESHOLD, j] = 1/DELTA_T
    V_i[V_i[:, j] > V_THRESHOLD, j] = 0


for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(t,V_i[i])
plt.show()

r_e = np.count_nonzero(spike_e, axis = 1).sum(axis = 0) / TIME_SPAN / N_E
r_i = np.count_nonzero(spike_i, axis = 1).sum(axis = 0) / TIME_SPAN / N_I
plt.hist(np.count_nonzero(spike_e, axis = 1) / TIME_SPAN, bins = range(0,200,5) ,label = 'excitatory')
plt.hist(np.count_nonzero(spike_i, axis = 1) / TIME_SPAN, bins = range(0,200,5), label = 'inhibitory')
plt.legend()
plt.show()

print('r_e = {:.2f}, r_i = {:.2f}'.format(r_e, r_i))

# %%
