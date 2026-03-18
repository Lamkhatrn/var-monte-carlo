import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ── Paramètres ──────────────────────────────────────────────────────────────
TICKERS      = ["^GSPC", "^FCHI"]   # S&P 500 + CAC 40
POIDS        = np.array([0.6, 0.4]) # 60% S&P, 40% CAC
SIMULATIONS  = 10_000
HORIZON      = 30                   # jours
CONFIANCE    = 0.95
CAPITAL      = 100_000              # euros

# ── 1. Données réelles ───────────────────────────────────────────────────────
print("Téléchargement des données...")
data = yf.download(TICKERS, period="2y", auto_adjust=True)["Close"]
data = data.dropna()

rendements = np.log(data / data.shift(1)).dropna()

# ── 2. Paramètres statistiques du portefeuille ───────────────────────────────
moyenne   = rendements.mean().values
cov       = rendements.cov().values

print(f"\nMatrice de corrélation :")
print(rendements.corr().round(3))

# ── 3. Simulation Monte Carlo ────────────────────────────────────────────────
np.random.seed(42)
simulations_rendements = np.random.multivariate_normal(
    moyenne, cov, (SIMULATIONS, HORIZON)
)

# Rendement cumulé sur l'horizon
rendements_cumules = simulations_rendements.sum(axis=1) @ POIDS
pertes = -CAPITAL * rendements_cumules

# ── 4. VaR et CVaR ───────────────────────────────────────────────────────────
var_95  = np.percentile(pertes, CONFIANCE * 100)
cvar_95 = pertes[pertes >= var_95].mean()

print(f"\n── Résultats ──────────────────────────────────────────")
print(f"Capital simulé       : {CAPITAL:,.0f} €")
print(f"VaR 95% à {HORIZON}j     : {var_95:,.0f} €")
print(f"CVaR 95% à {HORIZON}j    : {cvar_95:,.0f} €")
print(f"Perte max simulée    : {pertes.max():,.0f} €")

# ── 5. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Value at Risk — Simulation Monte Carlo", fontsize=14, fontweight="bold")

# Distribution des pertes
ax1 = axes[0]
ax1.hist(pertes, bins=80, color="#4C72B0", alpha=0.7, edgecolor="none")
ax1.axvline(var_95,  color="#E74C3C", linewidth=2, label=f"VaR 95% = {var_95:,.0f} €")
ax1.axvline(cvar_95, color="#E67E22", linewidth=2, linestyle="--",
            label=f"CVaR 95% = {cvar_95:,.0f} €")
ax1.set_xlabel("Perte simulée (€)")
ax1.set_ylabel("Fréquence")
ax1.set_title("Distribution des pertes sur 30 jours")
ax1.legend()

# Quelques trajectoires simulées
ax2 = axes[1]
prix_trajectoires = CAPITAL * np.exp(
    simulations_rendements[:200, :, :] @ POIDS[:, None] * np.ones((1, HORIZON))
).cumprod(axis=1)

for i in range(200):
    ax2.plot(prix_trajectoires[i], alpha=0.05, color="#4C72B0", linewidth=0.8)
ax2.axhline(CAPITAL, color="black", linewidth=1.5, linestyle="--", label="Capital initial")
ax2.set_xlabel("Jours")
ax2.set_ylabel("Valeur du portefeuille (€)")
ax2.set_title("200 trajectoires simulées")
ax2.legend()

plt.tight_layout()
plt.savefig("var_monte_carlo.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGraphique sauvegardé : var_monte_carlo.png")