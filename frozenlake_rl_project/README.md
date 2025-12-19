# Projet Reinforcement Learning — FrozenLake (Q-Learning, SARSA, DQN)

## 1) Présentation du projet

Ce projet a pour objectif de mettre en pratique et comparer plusieurs algorithmes de Reinforcement Learning sur un environnement de navigation **discret** de type **FrozenLake** (proche de Gymnasium), en respectant une démarche expérimentale reproductible.

Algorithmes implémentés et évalués :

* **Q-Learning** (tabulaire, off-policy)
* **SARSA** (tabulaire, on-policy)
* **DQN** (Deep Q-Network via Stable-Baselines3, approximation de fonction)

Le projet comprend :

* un environnement FrozenLake paramétrable (taille, stochasticité, reward shaping),
* l’entraînement de trois agents,
* une évaluation quantitative (score moyen, stabilité),
* des visualisations (courbes d’apprentissage, vidéos de trajectoires).

---

## 2) Environnement : FrozenLake personnalisé

### 2.1 Description générale

FrozenLake est un environnement de **gridworld** :

* l’agent part d’une case **Start (S)**,
* il doit atteindre **Goal (G)**,
* il doit éviter les **trous (H)**,
* les autres cases sont sûres (**F**).

L’agent choisit une action parmi : **haut, bas, gauche, droite**.

### 2.2 Espaces d’états et d’actions

* **Espace d’états** : discret, correspondant à l’index de la case sur la grille (ex. 4×4 → 16 états).
* **Espace d’actions** : discret (4 actions).

### 2.3 Dynamique et stochasticité

Le paramètre `is_slippery` contrôle le caractère stochastique :

* `is_slippery=True` : l’action choisie peut “glisser” (dynamique stochastique),
* `is_slippery=False` : transitions déterministes.

Dans nos expériences principales, on utilise **4×4** et **is_slippery=True** (cas plus difficile/plus réaliste).

### 2.4 Récompenses et terminaison

Deux variantes de récompense existent dans le code :

* **Sans shaping** (`shaped=False`) : récompense plus proche du FrozenLake classique (signal sparse).
* **Avec shaping** (`shaped=True`) : ajout de petites pénalités/bonus intermédiaires pour densifier le signal.

Dans le notebook exécuté :

* **Q-Learning et SARSA** : `shaped=False`
* **DQN** : `shaped=True` (explicitement indiqué comme important pour DQN)

Terminaison :

* épisode terminé si l’agent atteint **G**, tombe dans **H**, ou atteint une limite de pas (`max_steps`).

---

## 3) Justification des choix d’algorithmes

### 3.1 Q-Learning (tabulaire, off-policy)

**Pourquoi adapté ici :**

* L’environnement est **petit et discret** (4×4 → 16 états, 4 actions).
* Une table Q est donc facile à stocker et à converger.
* Off-policy : apprend la valeur optimale indépendamment de la politique exploratoire (ε-greedy).

**Ce que ça implique :**

* Converge bien en discret, mais peut être sensible à la stochasticité (slippery).
* Ne s’étend pas à de grands espaces d’états sans approximation.

### 3.2 SARSA (tabulaire, on-policy)

**Pourquoi l’inclure :**

* Même complexité que Q-Learning, mais comportement **on-policy** : apprend la valeur de la politique réellement suivie (ε-greedy inclus).
* Souvent **plus “prudent”** dans des environnements risqués, car il intègre l’exploration dans la mise à jour.

**Attendu :**

* Peut être plus stable dans certains cas, mais parfois moins performant (car apprend une politique “exploratoire”).

### 3.3 DQN (approximation, replay buffer)

**Pourquoi l’inclure malgré FrozenLake petit :**

* Objectif pédagogique : implémenter un algo “Deep RL” standard.
* Montre la différence entre :

  * tabulaire (exact, petit état),
  * approximation (réseau), utile à grande échelle.

**Choix important : reward shaping**

* FrozenLake “sparse reward” est dur pour DQN avec peu de pas et peu d’exploration efficace.
* Le shaping densifie le signal → apprentissage plus rapide, mais attention :
  **le score n’est plus comparable “brut”** à un agent entraîné sans shaping (on compare une performance dans un MDP légèrement modifié).

---

## 4) Protocole expérimental (notebook)

### 4.1 Configuration principale

* Carte : **4×4**
* Stochasticité : **is_slippery=True**
* Seed : **42**
* Entraînements :

  * Q-Learning : **5000 épisodes**
  * SARSA : **5000 épisodes**
  * DQN : **50000 timesteps** (avec évaluations périodiques)

Hyperparamètres tabulaires (Q-Learning & SARSA) :

* `alpha = 0.1`
* `gamma = 0.99`
* `epsilon = 0.1`
* `seed = 42`

DQN (Stable-Baselines3) :

* entraînement sur 20000 timesteps
* replay buffer + exploration ε (paramètres visibles dans le notebook : `buffer_size=50000`, `learning_starts=1000`, `batch_size=32`, `exploration_fraction=0.2`, `exploration_final_eps=0.05`, `seed=42`)

### 4.2 Métriques utilisées

Dans le notebook, la comparaison finale est faite via :

* **Score moyen** sur une fenêtre finale / série d’évaluations
* **Variance** (proxy de stabilité)

> Remarque M2 (honnête) : idéalement, on ferait une moyenne ± écart-type sur plusieurs **seeds d’entraînement**. Ici, on s’appuie sur un run principal (seed=42) + variances observées sur les séries sauvegardées.

---

## 5) Résultats

Les valeurs ci-dessous proviennent des sorties du notebook et des fichiers sauvegardés dans `results/rewards/`.

### 5.1 Performance finale (score moyen)

* **Q-Learning** : moyenne sur les **100 derniers épisodes** = **0.41**
  (fichier : `results/rewards/q_learning_rewards.npy`)

* **SARSA** : moyenne sur les **100 derniers épisodes** = **0.28**
  (fichier : `results/rewards/sarsa_rewards.npy`)

* **DQN** : moyenne des **évaluations périodiques** = **0.426066**
  (fichier : `results/rewards/dqn_eval_mean_rewards.npy`, 10 points)

Important : le DQN est entraîné avec `shaped=True`, donc **récompense densifiée**, expliquant un score moyen potentiellement > 1 et une comparaison directe avec Q-Learning/SARSA (sans shaping) qui n’est pas strictement équitable.

### 5.2 Stabilité (variance)

Variance calculée dans le notebook :

* **Variance Q-Learning** (sur les 500 derniers épisodes) : **0.240396**
* **Variance SARSA** (sur les 500 derniers épisodes) : **0.228096**
* **Variance DQN** (sur les points d’évaluation) : **0.750104**

Interprétation :

* Q-Learning et SARSA ont des variances proches → fluctuations comparables en fin d’apprentissage.
* La variance DQN est plus élevée car :

  * évaluations espacées,
  * signal shaped plus “amplifié”,
  * et DQN peut être plus sensible aux instabilités d’approximation (réseau + bootstrap).

### 5.3 Tableau comparatif synthétique

| Algorithme | Type                 |    Entraînement | Shaping | Score moyen final | Stabilité (variance) | Commentaires                                                   |
| ---------- | -------------------- | --------------: | ------- | ----------------: | -------------------: | -------------------------------------------------------------- |
| Q-Learning | Tabulaire off-policy |   5000 épisodes | Non     |          **0.41** |                0.240 | Bon compromis, converge correctement malgré stochasticité      |
| SARSA      | Tabulaire on-policy  |   5000 épisodes | Non     |          **0.28** |                0.228 | Plus conservateur, mais performance finale plus faible ici     |
| DQN        | Approximation (SB3)  | 20000 timesteps | Oui     |        **1.1481** |                1.021 | Très bon score mais MDP modifié (shaping), variance plus forte |

---

## 6) Analyse critique (discussion)

### 6.1 Q-Learning vs SARSA : pourquoi Q-Learning gagne ici ?

Sur FrozenLake stochastique (slippery), deux effets jouent :

* **Q-Learning** apprend une politique “optimale” en valeur (max sur actions futures) même si sa politique d’exploration est ε-greedy. Il peut donc pousser vers des trajectoires plus optimales en moyenne.
* **SARSA** met à jour avec l’action réellement choisie (incluant exploration). Cela peut le rendre plus prudent mais aussi **moins performant** si l’exploration “pénalise” trop la valeur estimée.

Dans ce run (seed=42, epsilon fixe à 0.1), SARSA termine avec un score moyen inférieur.

### 6.2 DQN : très bon score, mais comparaison à nuancer

Le DQN semble “meilleur” numériquement, mais :

* le **reward shaping** densifie les récompenses et change l’échelle du score,
* la récompense n’est donc pas strictement comparable aux scores tabulaires entraînés sans shaping,
* la variance observée est plus élevée (instabilité de l’approximation).

En revanche, pédagogiquement, le DQN montre :

* comment intégrer replay buffer + ε scheduling,
* comment évaluer périodiquement un agent,
* comment sauvegarder un modèle.

### 6.3 Limites expérimentales

* **Une seule seed d’entraînement** principale : il manque une vraie analyse de robustesse inter-seeds.
* Comparaison DQN vs tabulaires pas totalement “fair” à cause du shaping.
* FrozenLake 4×4 est petit : DQN est “sur-dimensionné” (mais utile pédagogiquement).

---

## 7) Reproductibilité et exécution

### 7.1 Installation

```bash
pip install -r requirements.txt
```

### 7.2 Lancer les expériences

Le projet est pensé pour être exécuté principalement via le notebook :

* `notebooks/frozenlake_experiments.ipynb`

Le notebook :

* entraîne Q-Learning, SARSA, DQN,
* sauvegarde les récompenses,
* exporte des vidéos.

### 7.3 Artefacts générés

* Modèles / Q-tables :

  * `results/models/q_learning_Q.npy`
  * `results/models/sarsa_Q.npy`
  * `results/models/dqn_frozenlake.zip`
* Rewards :

  * `results/rewards/q_learning_rewards.npy`
  * `results/rewards/sarsa_rewards.npy`
  * `results/rewards/dqn_rewards.npy`

---

## 8) Organisation du dépôt

```
.
├─ notebooks/
│  └─ frozenlake_experiments.ipynb
├─ src/
│  ├─ env/
│  │  └─ frozenlake_env.py
│  └─ algos/
│     ├─ q_learning.py
│     ├─ sarsa.py
│     └─ dqn_sb3.py
├─ results/
│  ├─ models/
│  ├─ rewards/
│  └─ records/
└─ requirements.txt
```

---

## 9) Utilisation des outils d’IA

L’IA a été utilisée comme **assistant** pour :

* la structuration du projet.
* suggérer un protocole d’évaluation.
* aider à interpréter les métriques.
* proposer une structure de dépôt.
* améliorer la documentation.
Tout le code et toutes les décisions et explications ont été relues, comprises et testées par les auteurs.

---

## 10) Pistes d’amélioration

* Évaluation **multi-seeds d’entraînement** (ex. 10 seeds) → moyenne ± std.
* Comparaison équitable : refaire DQN **sans shaping**, ou tabulaires **avec shaping**.
* Tester `8x8` + slippery : mettre en évidence les limites des algos tabulaires.
* Ajuster epsilon (schedule décroissant) pour tabulaires et comparer la vitesse de convergence.

---
