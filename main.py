import numpy as np

# On augmente le nombre de simulations pour être plus précis
simulations = 1000
jours = 30
prix_initial = 100
volatilité = 0.20
dt = 1/252

# 1. Simuler les rendements finaux après 30 jours
rendements_finaux = np.random.normal(0, volatilité * np.sqrt(jours*dt), simulations)
prix_finaux = prix_initial * (1 + rendements_finaux)

# 2. Calculer la VaR à 95% (percentile 5)
# On cherche le prix en dessous duquel se trouvent seulement 5% des simulations
seuil_5_pourcent = np.percentile(prix_finaux, 5)
var_95 = prix_initial - seuil_5_pourcent

print(f"Le prix initial était de {prix_initial}€")
print(f"Dans 95% des cas, mon prix ne descendra pas en dessous de : {seuil_5_pourcent:.2f}€")
print(f"Ma Value at Risk (VaR 95%) est donc de : {var_95:.2f}€")