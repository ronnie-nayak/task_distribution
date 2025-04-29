import matplotlib.pyplot as plt
import numpy as np
from gaNsga import nsga
from gwoMogwo import mogwo
from acoMoaco import multi_objective_aco
from psoMopso import multi_objective_pso
from saMosa import multi_objective_sa

def main():
    folder = "task40"
    
    nsgaFitness = nsga(folder)
    nsgaTimes = [f[0] for f in nsgaFitness]
    nsgaCosts = [f[1] for f in nsgaFitness]

    gwoFitnesses = mogwo(folder)
    gwoTimes = [f[0] for f in gwoFitnesses]
    gwoCosts = [f[1] for f in gwoFitnesses]

    acoArchive, _ = multi_objective_aco(folder)
    acoParetoFront = np.array([p['fitness'] for p in acoArchive])

    psoArchive, _ = multi_objective_pso(folder)
    psoParetoFront = np.array([p['fitness'] for p in psoArchive])

    saArchive, _ = multi_objective_sa(folder)
    saParetoFront = np.array([p['fitness'] for p in saArchive])

    plt.figure(figsize=(10, 6))
    plt.scatter(nsgaTimes, nsgaCosts, color='blue', s=40)
    plt.scatter(gwoTimes, gwoCosts, color='green', s=40)
    plt.scatter(acoParetoFront[:, 0], acoParetoFront[:, 1], color='red', s=40)
    plt.scatter(psoParetoFront[:, 0], psoParetoFront[:, 1], color='purple', s=40)
    plt.scatter(saParetoFront[:, 0], saParetoFront[:, 1], color='orange', s=40)
    plt.xlabel("Execution Time")
    plt.ylabel("Execution Cost")
    plt.title("NSGA-II Pareto Front")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
