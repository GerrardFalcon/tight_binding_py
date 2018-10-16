import numpy as np
import matplotlib.pyplot as plt
import sisl as si

def onsite_y(y0, y, y_length, relax):
    """
    y0: center of channel
    y: coordinates to apply the potential too
    y_length: y0 - y_length / 2 -- y0 + y_length / 2 is the channel position
    """
    tanhL = np.tanh( (y - y0 - y_length * 0.5) / relax)
    tanhR = np.tanh( - (y - y0 + y_length * 0.5) / relax)

    return 0.5 * (tanhL + tanhR)
    
def potential(H, ia, idxs, idxs_xyz=None):
    # Retrieve all atoms close to ia
    idx = H.geometry.close(ia, R=[0.1, 1.44], idx=idxs, idx_xyz=idxs_xyz)
    
    H[ia, idx[0]] = onsite_y(H.geometry.center(what='cell')[1], H.geometry.xyz[idx[0], 1], 50, 5)
    H[ia, idx[1]] = -2.7

graphene = si.geom.graphene(orthogonal=True).tile(200, 1)
H = si.Hamiltonian(graphene)
H.construct(potential)

# Extract the diagonal (on-site terms of the Hamiltonian)
diag = H.Hk().diagonal()
plt.plot(H.geometry.xyz[:, 1], diag)
plt.xlabel('y [Ang]')
plt.ylabel('Onsite [eV]')
plt.show()
