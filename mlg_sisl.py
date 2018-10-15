import sisl
import matplotlib.pyplot as plt

bond = 1.42
# Construct the atom with the appropriate orbital range
# Note the 0.01 which is for numerical accuracy.
C = sisl.Atom(6, R = bond + 0.01)
# Create graphene unit-cell
gr = sisl.geom.graphene(bond, C)

print(gr)

# Create the tight-binding Hamiltonian
H = sisl.Hamiltonian(gr)
R = [0.1 * bond, bond + 0.01]

for ia in gr:
    idx_a = gr.close(ia, R)
    # On-site
    H[ia, idx_a[0]] = 0.
    # Nearest neighbour hopping
    H[ia, idx_a[1]] = -2.7

# Calculate eigenvalues at K-point

band = sisl.BandStructure(H, [[0, 0, 0], [0, 0.5, 0],
                  [1/3, 2/3, 0], [0, 0, 0]],
              400, [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])


bs = band.asarray().eigh()

lk, kt, kl = band.lineark(True)
plt.xticks(kt, kl)
plt.xlim(0, lk[-1])
plt.ylim([-3, 3])
plt.ylabel('$E-E_F$ [eV]')
for bk in bs.T:
    plt.plot(lk, bk)

plt.show()