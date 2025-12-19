# Projet de Reinforcement Learning — FrozenLake avancé

## 1. Présentation générale du projet

Ce projet s’inscrit dans le cadre du cours de Reinforcement Learning (niveau M2). Il vise à mettre en pratique, comparer et analyser plusieurs algorithmes d’apprentissage par renforcement sur un environnement discret classique : **FrozenLake-v1**, dans une version enrichie (stochasticité, reward shaping).

L’objectif principal est d’étudier le comportement et les performances d’algorithmes **tabulaires** et **deep reinforcement learning** dans un environnement stochastique à récompense sparse.

Algorithmes étudiés :

* Q-Learning
* SARSA
* DQN (Deep Q-Network)

---

## 2. Description de l’environnement

### 2.1 Environnement choisi

L’environnement utilisé est **FrozenLake-v1** (bibliothèque Gymnasium).

FrozenLake est une grille représentant un lac gelé où un agent doit atteindre une case objectif sans tomber dans des trous.

### 2.2 États

* Chaque état correspond à la **position de l’agent** sur la grille.
* L’espace d’états est discret :

  * 4x4 → 16 états

### 2.3 Actions

L’agent dispose de **4 actions discrètes** :

* 0 : Gauche
* 1 : Bas
* 2 : Droite
* 3 : Haut

### 2.4 Dynamique et stochasticité

* L’environnement est configuré avec `is_slippery=True`.
* Une action peut mener à une direction inattendue, ce qui introduit de la stochasticité.

### 2.5 Récompenses

L’environnement original est à récompense sparse (1 uniquement à l’objectif).

Afin de faciliter l’apprentissage, un **reward shaping** a été introduit :

* Petite pénalité à chaque pas
* Pénalité supplémentaire en cas de chute dans un trou
* Récompense positive à l’atteinte de l’objectif

Ce choix permet une convergence plus rapide et une meilleure stabilité, notamment pour DQN.

### 2.6 Conditions de terminaison

Un épisode se termine lorsque :

* l’agent atteint la case objectif (succès)
* l’agent tombe dans un trou (échec)
* le nombre maximal d’étapes est atteint

---

## 3. Algorithmes implémentés

### 3.1 Q-Learning

* Algorithme **off-policy**
* Mise à jour basée sur la meilleure action future
* Sensible à la stochasticité

Hyperparamètres principaux :

* α (learning rate)
* γ (discount factor)
* ε (ε-greedy exploration)

### 3.2 SARSA

* Algorithme **on-policy**
* Mise à jour basée sur l’action réellement suivie
* Plus conservateur et plus stable que Q-Learning

### 3.3 DQN

* Approche **Deep Reinforcement Learning**
* Approximation de la fonction Q par un réseau de neurones
* Implémentation via Stable-Baselines3

Caractéristiques :

* Replay buffer
* Réseau cible
* Entraînement sur plusieurs timesteps

---

## 4. Méthodologie expérimentale

* Environnement identique pour tous les algorithmes
* Même seed lorsque possible
* Entraînement sur un grand nombre d’épisodes / timesteps
* Évaluation basée sur plusieurs épisodes sans exploration

Métriques utilisées :

* Récompense moyenne
* Vitesse de convergence
* Stabilité (variance)
* Coût computationnel

---

## 5. Résultats et comparaison

### 5.1 Courbes d’apprentissage

Les courbes montrent :

* Une convergence progressive pour Q-Learning et SARSA
* Une meilleure performance finale pour DQN

### 5.2 Tableau comparatif

| Algorithme | Type       | Récompense moyenne | Stabilité   | Coût calcul |
| ---------- | ---------- | ------------------ | ----------- | ----------- |
| Q-Learning | Off-policy | Élevée             | Moyenne     | Faible      |
| SARSA      | On-policy  | Moyenne            | Élevée      | Faible      |
| DQN        | Deep RL    | Très élevée        | Très élevée | Élevé       |

---

## 6. Analyse critique

* Les méthodes tabulaires sont efficaces sur des environnements discrets de petite taille.
* SARSA est plus robuste dans un environnement stochastique.
* DQN offre la meilleure généralisation mais nécessite davantage de ressources.
* Le reward shaping est crucial pour l’apprentissage dans FrozenLake.

### Limites

* Environnement relativement simple
* Hyperparamètres non optimisés exhaustivement
* Comparaison réalisée sur une configuration principale

---

## 7. Utilisation des outils d’IA

Des outils d’IA (ChatGPT) ont été utilisés pour :

* Aide à la structuration du projet
* Assistance à l’implémentation
* Vérification de la cohérence méthodologique

Tous les choix et le code ont été compris, analysés et validés par les auteurs.

---

## 8. Instructions d’exécution

### Installation

```bash
pip install -r requirements.txt
```

### Lancement

Ouvrir le notebook :

```bash
jupyter notebook notebooks/01_frozenlake_experiments.ipynb
```

Toutes les expériences, visualisations et analyses sont contenues dans ce notebook.

---

## 9. Auteurs

Projet réalisé dans le cadre du Master 2 Intelligence Artificielle.
