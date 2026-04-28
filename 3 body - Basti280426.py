import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.text as text

'''
Was man noch hinzufügen/schöner machen könnte/sollte:
- 3D ? Einfacher Wechsel zwischen 2D und 3D
-Einfacheres System für nicht 2,3 sondern n Körper -> classes? Zufallsprinzip bei q und p
-Visualisierung nur des Potentials von 2D n-Körper-Problemen in 3D Chart?
-Interface zum Einstellen/Ordentliches Front-End
-H,V,T live update legende
-Custom Vorlagen (z.B. Typisches Erde - Sonne, Mond - Erde - Sonne, Sonnensystem, 100 Körper?)
-Bessere/Zugängliche Lösung für q,p,m Anfangsbedingungen
-Spezialfälle für kleines/großer r_ij --> dt ändern
-elastische/inelastische Stöße ? Vllt gar nicht so schwer, wenn man eh schon mit impulsen rechnet...
'''

G = 1 # Gravitationskonstante
G = 6.674 * 10**(-11) #tatsächliche Gravitationskonstante

D = 2 # Dimensionen
B = 3 # Bodies

class solar_system():
    m1 = 1.989 * 10**30 # Masse der Sonne [kg]
    m2 = 5.972 * 10**24 # Masse der Erde [kg]
    m3 = 7.35 * 10**22  # Masse des Mondes [kg]
    q1 = np.array([0., 0.]) # Heute mal ein heliozentrisches Weltbild
    p1 = np.array([0., -29.8 * 10 **3 * m2 - 30.822 * 10 ** 3 * m3]) # Sonne bewegt sich nicht, genau deswegen braucht es Impulserhaltung 
    #p1 = np.array([m1 * 6464, 0.]) # Das hier sieht ganz cool aus.
    q2 = np.array([149.6 * 10**9, 0.]) #Position der Erde [m]
    p2 = np.array([0., 29.8 * 10 **3 * m2]) #Geschwindigkeit und Masse der Erde um Sonne (orthogonal) [m/s]
    q3 = np.array([149.6 * 10**9 + 384.4 * 10**6, 0.]) # Position Mond bei Sonne - Mond - Erde [m]
    p3 = np.array([0., 30.822 * 10 ** 3 * m3]) # Geschwindigkeit Mond mit zur Erde rel. geringer Geschwindigkeit
sol = solar_system()


class test():
    m1 = 10
    m2 = 500
    q1 = np.array([10., 0.])
    p1 = np.array([0., 0.])
    q2 = np.array([0., 100.])
    p2 = np.array([2., 0.])

q = np.array([sol.q1, sol.q2, sol.q3])
p = np.array([sol.p1, sol.p2, sol.p3])
m = np.array([sol.m1, sol.m2, sol.m3])

#q = np.array([test.q1, test.q2, test.q3])
#p = np.array([test.p1, test.p2, test.p3])
#m = np.array([test.m1, test.m2, test.m3])



# -- T -- #
T = 0.5 * np.sum(np.linalg.norm(p, axis=1)**2 / m)

# -- V -- #
V=0
for i in range(len(q)):
    for j in range(i +1 , len(q)):
        r_ij = np.linalg.norm(q[i]-q[j])+0.01**2
        V+= -G * m[i] * m[j] /r_ij

# -- H -- #
H=T+V

print(T)
print(V)
print(H)

t_steps =20000 # ca. 2,25 Jahre
dt = 3600.


history = np.zeros((t_steps, len(q), D))
#historyH = np.zeros((t_steps, 1))

def acceleration(q, m):
    q_tt = np.zeros_like(q)
    for i in range(len(q)):
        for j in range(i+1 , len(q)):
            r = q[i] - q[j]
            epsilon = 0.01
            r_ij = np.linalg.norm(q[i]-q[j]) + epsilon**2 # damit man bei r_ij nicht durch 0 teilt
            q_tt[i] += -G * m[j] * r / r_ij**3 # auf i wirkt die masse von j
            q_tt[j] +=  G * m[i] * r / r_ij**3 # auf j wirkt die masse von i
    return q_tt

a = acceleration(q, m)
for step in range(t_steps):                 # Leapfrog (anscheinend stabiler als Euler und Runge-Kutta (long-term)), besonders gut bei Energieerhaltung (!)
    p += 0.5 * dt * (a * m[:, None])        # half kick
    q += dt * p / m[:, None]                # drift
    a = acceleration(q, m)                  # one force eval
    p += 0.5 * dt * (a * m[:, None])        # half kick

#    T = 0.5 * np.sum(np.linalg.norm(p, axis=1)**2 / m)
#    for i in range(len(q)):
#        for j in range(i +1 , len(q)):
#            r_ij = np.linalg.norm(q[i]-q[j])
#            V+= -G * m[i] * m[j] /r_ij
#    H=T+V

#    historyH[step] = H
    history[step] = q

# -- Visualisierung -- #

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(history[:, :, 0].min() * 1.05 , history[:, :, 0].max() * 1.05)
ax.set_ylim(history[:, :, 1].min() *1.05 , history[:, :, 1].max() * 1.05)
ax.set_aspect('equal')

trail1, = ax.plot([], [], '-', lw=0.7, alpha=0.5)
trail2, = ax.plot([], [], '-', lw=0.7, alpha=0.5)
trail3, = ax.plot([], [], '-', lw=0.7, alpha=0.5)
dot1, = ax.plot([], [], 'o', ms=12)
dot2, = ax.plot([], [], 'o', ms=6)
dot3, = ax.plot([], [], 'o', ms=4)

steps = 2 # Je kleiner die steps, desto langsamer die Animation

def update(frame):
    i = frame * steps
    trail1.set_data(history[:i, 0, 0], history[:i, 0, 1])
    trail2.set_data(history[:i, 1, 0], history[:i, 1, 1])
    trail3.set_data(history[:i, 2, 0], history[:i, 2, 1,])
    dot1.set_data([history[i, 0, 0]], [history[i, 0, 1]])
    dot2.set_data([history[i, 1, 0]], [history[i, 1, 1]])
    dot3.set_data([history[i, 2, 0]], [history[i, 2, 1]])
    return trail1, trail2, trail3, dot1, dot2, dot3

ani = animation.FuncAnimation(
    fig, update, frames=len(history) // steps,
    interval=20, blit=True, repeat = False
)
plt.xlabel("x-Koordinate")
plt.ylabel("y-Koordinate")
plt.show()

