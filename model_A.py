# %%
import numpy as np
import matplotlib.pyplot as plt

# Initialisation of parameters
TIME_SPAN = 100 # Unit: ms
DELTA_T = 0.1 # Unit: ms
TAU = 20 # Time constant, unit: ms
TAU_RP = 2 # Refractory time, unit: ms
DELAY = 2 # Unit: ms
D_B = int(DELAY/DELTA_T) # delay bins

N_E = 10000 # Excitatory population
GAMMA = 0.25
N_I = int(GAMMA * N_E) # Inhibitory population
N_O = N_E # Outside excitatory population
EPSILON = 0.1 
C_E = int(N_E * EPSILON)
C_I = int(N_I * EPSILON)
C_O = C_E

CONST_G = 7 # Ratio = J_I / J_E
J_E = 0.2
J_I = -CONST_G * J_E
THETA = 20 # Threshold, unit: mv
V_R = 10 # Reset potential, unit: mv
RATE_THR = THETA / (J_E * C_E *TAU) # Firing rate threshold, unit: ms-1
RATE_O = 3.5 * RATE_THR # Firing threshold, unit: ms-1


# Initialization of populations of neurons 
t = np.arange(0, TIME_SPAN, DELTA_T)

V_e = V_R * np.ones((N_E, len(t)))
spike_e = np.zeros((N_E, len(t)))

V_i = V_R * np.ones((N_I, len(t)))
spike_i = np.zeros((N_I, len(t)))

spike_o = 1/DELTA_T * np.random.binomial(1, RATE_O / 10, size = (N_O, len(t)))

plt.hist(np.count_nonzero(spike_o, axis = 1))
plt.show()

# Generation of connections between neurons
c_ee = np.random.binomial(1, EPSILON, size = (N_E, N_E))
c_ei = np.random.binomial(1, EPSILON, size = (N_E, N_I))
c_eo = np.random.binomial(1, EPSILON, size = (N_E, N_O))

c_ie = np.random.binomial(1, EPSILON, size = (N_I, N_E))
c_ii = np.random.binomial(1, EPSILON, size = (N_I, N_I))
c_io = np.random.binomial(1, EPSILON, size = (N_I, N_O))

ri_e = np.zeros((N_E, len(t)))
ri_i = np.zeros((N_I, len(t)))

plt.hist(np.count_nonzero(c_ee, axis = 1))
plt.hist(np.count_nonzero(c_ei, axis = 1))
plt.hist(np.count_nonzero(c_eo, axis = 1))
plt.show()

plt.hist(np.count_nonzero(c_ie, axis = 1))
plt.hist(np.count_nonzero(c_ii, axis = 1))
plt.hist(np.count_nonzero(c_io, axis = 1))
plt.show()

# Integration
for j in range(D_B, len(t)):
    ri_e[:, j-1] = TAU * (J_E * np.dot(c_ee, spike_e[:, j-D_B-1]) + J_E * np.dot(c_eo, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ei, spike_i[:, j-D_B-1]))
    ri_i[:, j-1] = TAU * (J_E * np.dot(c_ie, spike_e[:, j-D_B-1]) + J_E * np.dot(c_io, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ii, spike_i[:, j-D_B-1]))
    V_e[:, j] = V_e[:, j-1] + DELTA_T/TAU * (-V_e[:, j-1] + ri_e[:, j-1]) 
    V_i[:, j] = V_i[:, j-1] + DELTA_T/TAU * (-V_i[:, j-1] + ri_i[:, j-1]) 

    aim_e = np.logical_and(V_e[:, j] > THETA, np.logical_not(np.any(spike_e[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
    spike_e[aim_e, j] = 1/DELTA_T
    V_e[aim_e, j] = V_R

    aim_i = np.logical_and(V_i[:, j] > THETA, np.logical_not(np.any(spike_i[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
    spike_i[aim_i, j] = 1/DELTA_T
    V_i[aim_i, j] = V_R

for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(t,V_i[i])
plt.show()

for i in range(0,5):
    plt.subplot(5,1,i+1)
    plt.plot(t,V_e[i])
plt.show()

plt.hist(np.count_nonzero(spike_e, axis = 1))
plt.hist(np.count_nonzero(spike_i, axis = 1))
plt.show()
print('RATE_E = {:.2f} Hz, RATE_I = {:.2f} Hz'.format(np.mean(np.count_nonzero(spike_e, axis = 1))/TIME_SPAN*1000, np.mean(np.count_nonzero(spike_i, axis = 1))/TIME_SPAN*1000))




# %% Fully sychronized network
import numpy as np
import matplotlib.pyplot as plt

# Initialisation of parameters
TIME_SPAN = 100 # Unit: ms
DELTA_T = 0.1 # Unit: ms
TAU = 20 # Time constant, unit: ms
TAU_RP = 2 # Refractory time, unit: ms
DELAY = 1.5 # Unit: ms
D_B = int(DELAY/DELTA_T) # delay bins

N_E = 10000 # Excitatory population
GAMMA = 0.25
N_I = int(GAMMA * N_E) # Inhibitory population
N_O = N_E # Outside excitatory population
EPSILON = 0.1 
C_E = int(N_E * EPSILON)
C_I = int(N_I * EPSILON)
C_O = C_E

CONST_G = 3 # Ratio = J_I / J_E
J_E = 0.1
J_I = -CONST_G * J_E
THETA = 20 # Threshold, unit: mv
V_R = 10 # Reset potential, unit: mv
RATE_THR = THETA / (J_E * C_E *TAU) # Firing rate threshold, unit: ms-1
RATE_O = 2 * RATE_THR # Firing threshold, unit: ms-1


# Initialization of populations of neurons 
t = np.arange(0, TIME_SPAN, DELTA_T)

V_e = V_R * np.ones((N_E, len(t)))
spike_e = np.zeros((N_E, len(t)))

V_i = V_R * np.ones((N_I, len(t)))
spike_i = np.zeros((N_I, len(t)))

spike_o = 1/DELTA_T * np.random.binomial(1, RATE_O / 10, size = (N_O, len(t)))

plt.hist(np.count_nonzero(spike_o, axis = 1))
plt.show()

# Generation of connections between neurons
c_ee = np.random.binomial(1, EPSILON, size = (N_E, N_E))
c_ei = np.random.binomial(1, EPSILON, size = (N_E, N_I))
c_eo = np.random.binomial(1, EPSILON, size = (N_E, N_O))

c_ie = np.random.binomial(1, EPSILON, size = (N_I, N_E))
c_ii = np.random.binomial(1, EPSILON, size = (N_I, N_I))
c_io = np.random.binomial(1, EPSILON, size = (N_I, N_O))

ri_e = np.zeros((N_E, len(t)))
ri_i = np.zeros((N_I, len(t)))


# Integration
for j in range(D_B, len(t)):
    print('\rCurrent time is {:.1f} ms / {:.1f} ms'.format(j*0.1+DELTA_T, TIME_SPAN), end = ' ')
    ri_e[:, j-1] = TAU * (J_E * np.dot(c_ee, spike_e[:, j-D_B-1]) + J_E * np.dot(c_eo, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ei, spike_i[:, j-D_B-1]))
    ri_i[:, j-1] = TAU * (J_E * np.dot(c_ie, spike_e[:, j-D_B-1]) + J_E * np.dot(c_io, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ii, spike_i[:, j-D_B-1]))
    V_e[:, j] = V_e[:, j-1] + DELTA_T/TAU * (-V_e[:, j-1] + ri_e[:, j-1]) 
    V_i[:, j] = V_i[:, j-1] + DELTA_T/TAU * (-V_i[:, j-1] + ri_i[:, j-1]) 

    if j >= int(TAU_RP/DELTA_T):
        aim_e = np.logical_and(V_e[:, j] > THETA, np.logical_not(np.any(spike_e[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = np.logical_and(V_i[:, j] > THETA, np.logical_not(np.any(spike_i[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R
    else:
        aim_e = V_e[:, j] > THETA
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = V_i[:, j] > THETA
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R

#%% Result plot
choice = np.random.randint(low = 0, high = N_E + N_I -1, size = 50)
spike = np.zeros((N_E+N_I, len(t)))
spike[0:N_E, :] = spike_e
spike[N_E:N_E+N_I, :] = spike_i
sample = spike[choice, :]
sample_re = (np.linspace(2, 51, num = 50) * sample.T).T * DELTA_T

plt.subplot(2,1,1)
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
ax.axes.set_ylim(bottom = 1.5, top = 53)
for i in range(0,50):
    plt.scatter(t[400:], sample_re[i, 400:], marker = '|', color = 'k')
plt.subplot(2,1,2)
plt.plot(t[400:], np.count_nonzero(spike, axis = 0)[400:])
plt.show()

# %% Fast oscillation of global activity
import numpy as np
import matplotlib.pyplot as plt

# Initialisation of parameters
TIME_SPAN = 100 # Unit: ms
DELTA_T = 0.1 # Unit: ms
TAU = 20 # Time constant, unit: ms
TAU_RP = 2 # Refractory time, unit: ms
DELAY = 1.5 # Unit: ms
D_B = int(DELAY/DELTA_T) # delay bins

N_E = 10000 # Excitatory population
GAMMA = 0.25
N_I = int(GAMMA * N_E) # Inhibitory population
N_O = N_E # Outside excitatory population
EPSILON = 0.1 
C_E = int(N_E * EPSILON)
C_I = int(N_I * EPSILON)
C_O = C_E

CONST_G = 6 # Ratio = J_I / J_E
J_E = 0.1
J_I = -CONST_G * J_E
THETA = 20 # Threshold, unit: mv
V_R = 10 # Reset potential, unit: mv
RATE_THR = THETA / (J_E * C_E *TAU) # Firing rate threshold, unit: ms-1
RATE_O = 4 * RATE_THR # Firing threshold, unit: ms-1


# Initialization of populations of neurons 
t = np.arange(0, TIME_SPAN, DELTA_T)

V_e = V_R * np.ones((N_E, len(t)))
spike_e = np.zeros((N_E, len(t)))

V_i = V_R * np.ones((N_I, len(t)))
spike_i = np.zeros((N_I, len(t)))

spike_o = 1/DELTA_T * np.random.binomial(1, RATE_O / 10, size = (N_O, len(t)))

plt.hist(np.count_nonzero(spike_o, axis = 1))
plt.show()

# Generation of connections between neurons
c_ee = np.random.binomial(1, EPSILON, size = (N_E, N_E))
c_ei = np.random.binomial(1, EPSILON, size = (N_E, N_I))
c_eo = np.random.binomial(1, EPSILON, size = (N_E, N_O))

c_ie = np.random.binomial(1, EPSILON, size = (N_I, N_E))
c_ii = np.random.binomial(1, EPSILON, size = (N_I, N_I))
c_io = np.random.binomial(1, EPSILON, size = (N_I, N_O))

ri_e = np.zeros((N_E, len(t)))
ri_i = np.zeros((N_I, len(t)))


# Integration
for j in range(D_B, len(t)):
    print('\rCurrent time is {:.1f} ms / {:.1f} ms'.format(j*0.1+DELTA_T, TIME_SPAN), end = ' ')
    ri_e[:, j-1] = TAU * (J_E * np.dot(c_ee, spike_e[:, j-D_B-1]) + J_E * np.dot(c_eo, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ei, spike_i[:, j-D_B-1]))
    ri_i[:, j-1] = TAU * (J_E * np.dot(c_ie, spike_e[:, j-D_B-1]) + J_E * np.dot(c_io, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ii, spike_i[:, j-D_B-1]))
    V_e[:, j] = V_e[:, j-1] + DELTA_T/TAU * (-V_e[:, j-1] + ri_e[:, j-1]) 
    V_i[:, j] = V_i[:, j-1] + DELTA_T/TAU * (-V_i[:, j-1] + ri_i[:, j-1]) 

    if j >= int(TAU_RP/DELTA_T):
        aim_e = np.logical_and(V_e[:, j] > THETA, np.logical_not(np.any(spike_e[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = np.logical_and(V_i[:, j] > THETA, np.logical_not(np.any(spike_i[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R
    else:
        aim_e = V_e[:, j] > THETA
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = V_i[:, j] > THETA
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R

# %% Result plot
choice = np.random.randint(low = 0, high = N_E + N_I -1, size = 50)
spike = np.zeros((N_E+N_I, len(t)))
spike[0:N_E, :] = spike_e
spike[N_E:N_E+N_I, :] = spike_i
sample = spike[choice, :]
sample_re = (np.linspace(2, 51, num = 50) * sample.T).T * DELTA_T

plt.subplot(2,1,1)
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
ax.axes.set_ylim(bottom = 1.5, top = 53)
for i in range(0,50):
    plt.scatter(t[400:], sample_re[i, 400:], marker = '|', color = 'k')
plt.subplot(2,1,2)
plt.plot(t[400:], np.count_nonzero(spike, axis = 0)[400:])
plt.show()


#%% Stationary global activity
import numpy as np
import matplotlib.pyplot as plt

# Initialisation of parameters
TIME_SPAN = 100 # Unit: ms
DELTA_T = 0.1 # Unit: ms
TAU = 20 # Time constant, unit: ms
TAU_RP = 2 # Refractory time, unit: ms
DELAY = 1.5 # Unit: ms
D_B = int(DELAY/DELTA_T) # delay bins

N_E = 10000 # Excitatory population
GAMMA = 0.25
N_I = int(GAMMA * N_E) # Inhibitory population
N_O = N_E # Outside excitatory population
EPSILON = 0.1 
C_E = int(N_E * EPSILON)
C_I = int(N_I * EPSILON)
C_O = C_E

CONST_G = 5 # Ratio = J_I / J_E
J_E = 0.1
J_I = -CONST_G * J_E
THETA = 20 # Threshold, unit: mv
V_R = 10 # Reset potential, unit: mv
RATE_THR = THETA / (J_E * C_E *TAU) # Firing rate threshold, unit: ms-1
RATE_O = 2 * RATE_THR # Firing threshold, unit: ms-1


# Initialization of populations of neurons 
t = np.arange(0, TIME_SPAN, DELTA_T)

V_e = V_R * np.ones((N_E, len(t)))
spike_e = np.zeros((N_E, len(t)))

V_i = V_R * np.ones((N_I, len(t)))
spike_i = np.zeros((N_I, len(t)))

spike_o = 1/DELTA_T * np.random.binomial(1, RATE_O / 10, size = (N_O, len(t)))

plt.hist(np.count_nonzero(spike_o, axis = 1))
plt.show()

# Generation of connections between neurons
c_ee = np.random.binomial(1, EPSILON, size = (N_E, N_E))
c_ei = np.random.binomial(1, EPSILON, size = (N_E, N_I))
c_eo = np.random.binomial(1, EPSILON, size = (N_E, N_O))

c_ie = np.random.binomial(1, EPSILON, size = (N_I, N_E))
c_ii = np.random.binomial(1, EPSILON, size = (N_I, N_I))
c_io = np.random.binomial(1, EPSILON, size = (N_I, N_O))

ri_e = np.zeros((N_E, len(t)))
ri_i = np.zeros((N_I, len(t)))


# Integration
for j in range(D_B, len(t)):
    print('\rCurrent time is {:.1f} ms / {:.1f} ms'.format(j*0.1+DELTA_T, TIME_SPAN), end = ' ')
    ri_e[:, j-1] = TAU * (J_E * np.dot(c_ee, spike_e[:, j-D_B-1]) + J_E * np.dot(c_eo, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ei, spike_i[:, j-D_B-1]))
    ri_i[:, j-1] = TAU * (J_E * np.dot(c_ie, spike_e[:, j-D_B-1]) + J_E * np.dot(c_io, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ii, spike_i[:, j-D_B-1]))
    V_e[:, j] = V_e[:, j-1] + DELTA_T/TAU * (-V_e[:, j-1] + ri_e[:, j-1]) 
    V_i[:, j] = V_i[:, j-1] + DELTA_T/TAU * (-V_i[:, j-1] + ri_i[:, j-1]) 

    if j >= int(TAU_RP/DELTA_T):
        aim_e = np.logical_and(V_e[:, j] > THETA, np.logical_not(np.any(spike_e[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = np.logical_and(V_i[:, j] > THETA, np.logical_not(np.any(spike_i[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R
    else:
        aim_e = V_e[:, j] > THETA
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = V_i[:, j] > THETA
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R

# %% Result plot
choice = np.random.randint(low = 0, high = N_E + N_I -1, size = 50)
spike = np.zeros((N_E+N_I, len(t)))
spike[0:N_E, :] = spike_e
spike[N_E:N_E+N_I, :] = spike_i
sample = spike[choice, :]
sample_re = (np.linspace(2, 51, num = 50) * sample.T).T * DELTA_T

plt.subplot(2,1,1)
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
ax.axes.set_ylim(bottom = 1.5, top = 53)
for i in range(0,50):
    plt.scatter(t[400:], sample_re[i, 400:], marker = '|', color = 'k')
plt.subplot(2,1,2)
plt.plot(t[400:], np.count_nonzero(spike, axis = 0)[400:])
plt.show()


# %% Slow oscillation of the global activity
import numpy as np
import matplotlib.pyplot as plt

# Initialisation of parameters
TIME_SPAN = 300 # Unit: ms
DELTA_T = 0.1 # Unit: ms
TAU = 20 # Time constant, unit: ms
TAU_RP = 2 # Refractory time, unit: ms
DELAY = 1.5 # Unit: ms
D_B = int(DELAY/DELTA_T) # delay bins

N_E = 10000 # Excitatory population
GAMMA = 0.25
N_I = int(GAMMA * N_E) # Inhibitory population
N_O = N_E # Outside excitatory population
EPSILON = 0.1 
C_E = int(N_E * EPSILON)
C_I = int(N_I * EPSILON)
C_O = C_E

CONST_G = 4.5 # Ratio = J_I / J_E
J_E = 0.1
J_I = -CONST_G * J_E
THETA = 20 # Threshold, unit: mv
V_R = 10 # Reset potential, unit: mv
RATE_THR = THETA / (J_E * C_E *TAU) # Firing rate threshold, unit: ms-1
RATE_O = 0.9 * RATE_THR # Firing threshold, unit: ms-1


# Initialization of populations of neurons 
t = np.arange(0, TIME_SPAN, DELTA_T)

V_e = V_R * np.ones((N_E, len(t)))
spike_e = np.zeros((N_E, len(t)))

V_i = V_R * np.ones((N_I, len(t)))
spike_i = np.zeros((N_I, len(t)))

spike_o = 1/DELTA_T * np.random.binomial(1, RATE_O / 10, size = (N_O, len(t)))

plt.hist(np.count_nonzero(spike_o, axis = 1))
plt.show()

# Generation of connections between neurons
c_ee = np.random.binomial(1, EPSILON, size = (N_E, N_E))
c_ei = np.random.binomial(1, EPSILON, size = (N_E, N_I))
c_eo = np.random.binomial(1, EPSILON, size = (N_E, N_O))

c_ie = np.random.binomial(1, EPSILON, size = (N_I, N_E))
c_ii = np.random.binomial(1, EPSILON, size = (N_I, N_I))
c_io = np.random.binomial(1, EPSILON, size = (N_I, N_O))

ri_e = np.zeros((N_E, len(t)))
ri_i = np.zeros((N_I, len(t)))


# Integration
for j in range(D_B, len(t)):
    print('\rCurrent time is {:.1f} ms / {:.1f} ms'.format(j*0.1+DELTA_T, TIME_SPAN), end = ' ')
    ri_e[:, j-1] = TAU * (J_E * np.dot(c_ee, spike_e[:, j-D_B-1]) + J_E * np.dot(c_eo, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ei, spike_i[:, j-D_B-1]))
    ri_i[:, j-1] = TAU * (J_E * np.dot(c_ie, spike_e[:, j-D_B-1]) + J_E * np.dot(c_io, spike_o[:, j-D_B-1]) + J_I * np.dot(c_ii, spike_i[:, j-D_B-1]))
    V_e[:, j] = V_e[:, j-1] + DELTA_T/TAU * (-V_e[:, j-1] + ri_e[:, j-1]) 
    V_i[:, j] = V_i[:, j-1] + DELTA_T/TAU * (-V_i[:, j-1] + ri_i[:, j-1]) 

    if j >= int(TAU_RP/DELTA_T):
        aim_e = np.logical_and(V_e[:, j] > THETA, np.logical_not(np.any(spike_e[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = np.logical_and(V_i[:, j] > THETA, np.logical_not(np.any(spike_i[:, j-int(TAU_RP/DELTA_T):j-1]>0)))
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R
    else:
        aim_e = V_e[:, j] > THETA
        spike_e[aim_e, j] = 1/DELTA_T
        V_e[aim_e, j] = V_R

        aim_i = V_i[:, j] > THETA
        spike_i[aim_i, j] = 1/DELTA_T
        V_i[aim_i, j] = V_R

# %% Result plot
choice = np.random.randint(low = 0, high = N_E + N_I -1, size = 50)
spike = np.zeros((N_E+N_I, len(t)))
spike[0:N_E, :] = spike_e
spike[N_E:N_E+N_I, :] = spike_i
sample = spike[choice, :]
sample_re = (np.linspace(2, 51, num = 50) * sample.T).T * DELTA_T

plt.subplot(2,1,1)
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
ax.axes.set_ylim(bottom = 1.5, top = 53)
for i in range(0,50):
    plt.scatter(t[400:], sample_re[i, 400:], marker = '|', color = 'k')
plt.subplot(2,1,2)
plt.plot(t[400:], np.count_nonzero(spike, axis = 0)[400:])
plt.show()
# %%
