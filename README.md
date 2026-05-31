# Devoir de Deep learning
Nom de l'étudiant : DOSSOU KOKO Mahugnon Dieu-Donné Luc

# Titre du projet : Classification Cats vs Dogs : CNN From Scratch vs Transfer Learning (ResNet18)

## Objectif du projet

L'objectif de ce projet est de comparer les performances d'un réseau de neurones convolutif (CNN) entraîné **from scratch** avec celles d'un modèle pré-entraîné utilisant le **transfer learning** sur la tâche de classification d'images **Cats vs Dogs**.

Les expériences visent à analyser :
* la vitesse de convergence ;
* les performances de classification ;
* l'impact du choix de l'optimiseur ;
* l'apport du transfert d'apprentissage.

---

## Environnement

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Le projet a été développé avec Python et PyTorch.

Les expériences ont été réalisées sur Google Colab. Lorsque le GPU n'était plus disponible, les expériences ont été poursuivies sur CPU.
---

## Organisation des données

Le dataset Cats vs Dogs n'est pas inclus dans ce dépôt comme demandé par le devoir. Vous pouvez télécharger le jeu de données déjà structuré de cette manière ici : https://www.google.com/url?q=https%3A%2F%2Fs3.amazonaws.com%2Fcontent.udacity-data.com%2Fnd089%2FCat_Dog_data.zip. Il est déjà séparé en jeu d’entraînement et jeu de test.Après avoir téléchargé le dataset, placez-le dans le répertoire racine du projet avec l'organisation suivante :

```text
cnn-catsdogs/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── Cat_Dog_data/
    ├── train/
    │   ├── cat/
    │   └── dog/
    │
    └── test/
        ├── cat/
        └── dog/
```

Après avoir téléchargé le dataset Cats vs Dogs, placez le dossier Cat_Dog_data dans votre Google Drive à l'emplacement souhaité.

Dans Google Colab, montez Google Drive :

from google.colab import drive
drive.mount('/content/drive')

Puis adaptez le répertoire de travail :

import os
os.chdir('/content/drive/MyDrive/.../cnn_cat_dog_image_classification')

Le notebook utilise ensuite :

data_dir = "Cat_Dog_data"

Le dossier Cat_Dog_data doit donc être présent dans le répertoire de travail sélectionné.

Le dossier `train` est utilisé pour l'entraînement des modèles. cependant, 
Le dossier `test` est utilisé pour l'évaluation des performances.

---
# Commandes pour entraîner

## Expérience A : CNN entraîné From Scratch

### Architecture

Le modèle CNN est composé de :

* 3 blocs convolutionnels ;
* Batch Normalization après chaque convolution ;
* fonction d'activation ReLU ;
* MaxPooling ;
* Dropout pour limiter le surapprentissage ;
* couches entièrement connectées pour la classification finale.

### Hyperparamètres

| Paramètre     | Valeur                                    |
| ------------- | ----------------------------------------- |
| Batch Size    | 16                                        |
| Learning Rate | 0.001                                     |
| Epochs        | 10                                         |
| Loss Function | CrossEntropyLoss                          |
| Scheduler     | StepLR                                    |
| Dropout       | 0.25 (convolutions), 0.5 (classification) |

### Optimiseurs comparés

#### Adam

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

#### SGD

```python
torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)
```

Le meilleur modèle est sauvegardé automatiquement :

```text
best_model_adam.pth
best_model_sgd.pth
```

---

## Expérience B : Transfer Learning avec ResNet18

### Modèle utilisé

ResNet18 pré-entraîné sur ImageNet.

### Stratégie

* Gel des couches convolutionnelles pré-entraînées.
* Remplacement de la couche finale (`fc`) par une nouvelle tête de classification adaptée au problème Cats vs Dogs.
* Utilisation de Dropout pour la régularisation.
* Optimiseur Adam retenu après comparaison avec SGD.

### Hyperparamètres

| Paramètre     | Valeur           |
| ------------- | ---------------- |
| Batch Size    | 16               |
| Learning Rate | 0.001            |
| Epochs        | 10               |
| Loss Function | CrossEntropyLoss |
| Scheduler     | StepLR           |
| Optimiseur    | Adam             |

Le meilleur modèle est sauvegardé dans :

```text
best_model_resnet18_adam.pth
```

---

## Commande pour recharger un modèle sauvegardé

### CNN From Scratch

```python
model.load_state_dict(
    torch.load("best_model_adam.pth")
)
```

### ResNet18

```python
best_transfer_model.load_state_dict(
    torch.load("best_model_resnet18_adam.pth")
)
```
## Choix expérimentaux
## Dataset preparation
Le dataset Cats  vs Dogs a été chargé  avec "imageFolder". Les données d'entrainement ont été divisées en : 
- 80% pour l'apprentissage (Train)
- 20% pour la validation (Validation)
Un ensemble de test indépendant a été utilisé pour l'évaluation finale des modèles. 
les pour le test sont dans un autre dossier séparé.

### Taille des images (96 × 96)
Les images ont été redimensionnées à **96 × 96 pixels** afin de réduire le coût computationnel de l'entraînement tout en conservant suffisamment d'informations visuelles pour distinguer les chats des chiens.
Une taille plus grande (par exemple 128 × 128 ou 224 × 224) aurait augmenté significativement le temps d'entraînement, particulièrement lors de l'exécution sur CPU.

### Batch Size (16)
Un **batch size de 16** a été utilisé afin de trouver un compromis entre :
* la vitesse d'entraînement ;
* la consommation mémoire ;
* la stabilité de l'optimisation.

Cette valeur s'est révélée adaptée sur Google Colab, notamment lorsque les expériences ont dû être poursuivies sur CPU après la fin de la disponibilité du GPU.

### Nombre d'époques
Le nombre d'époques a été volontairement limité à 10 afin de maintenir des temps d'exécution raisonnables tout en permettant la comparaison des différentes approches (CNN from scratch, Adam vs SGD et Transfer Learning avec ResNet18).

### Choix de ResNet18
ResNet18 a été retenu pour l'expérience de transfert d'apprentissage car il offre un bon compromis entre performances et coût de calcul. Son architecture est suffisamment légère pour être entraînée sur CPU également tout en bénéficiant des connaissances acquises sur ImageNet.

---

## Résultats

### Comparaison Adam vs SGD

| Optimiseur | Loss        | Accuracy    | Precision   | Recall      |
| ---------- | ----------- | ----------- | ----------- | ----------- |
| Adam       | 0,4768      | 0,7632      | 0,8063      | 0,6928      |
| SGD        | 0,6004      | 0,6836      | 0,8637      | 0,4350      |

Adam a montré une convergence plus rapide et de meilleures performances que SGD sur le CNN entraîné from scratch.

### Comparaison CNN from scratch vs Transfer Learning

| Modèle                     | Loss        | Accuracy    | Precision   | Recall      |
| -------------------------- | ----------- | ----------- | ----------- | ----------- |
| CNN From Scratch           | 0,4768      | 0,7632      | 0,8063      | 0,6928      |
| ResNet18 Transfer Learning | 0,3023      | 0,8684      | 0,8847      | 0,8472      |

Le modèle ResNet18 pré-entraîné a montré une convergence plus rapide et de meilleures performances globales que le CNN entraîné from scratch.

La matrice de confusion du modele ResNet8 montre une bonne capacité de discrimination entre les classes "chat" et chien. Sur un total de 2500 images de test, 2171 ont été correctement classées, ce qui correspond à une exaltitude  de 86,84%. Les erreurs de classification restent limitées avec 191 chats confondus avec des chiens et 138 chiens confondus avec des chats. Les valeurs de précision et de rappel superieurs à 84% pour les deux classes confirment la robustesse du modèle et sa bonne capacité de généralisation.

Les courbes et graphs générées dans le notebook présentent :

* comparaison Adam vs SGD (Train Loss et Test Loss) ;
* comparaison Adam vs SGD (Train Accuracy et Test Accuracy) ;
* comparaison CNN From Scratch vs ResNet18 (Train Loss et Test Loss) ;
* comparaison CNN From Scratch vs ResNet18 (Train Accuracy et Test Accuracy) ;
* Matrix de confusion ResNet18
Les courbes sont dans le dossier image du dépot Github
---

## Analyse

Les résultats montrent que l'optimiseur Adam converge plus rapidement que SGD sur le CNN entraîné from scratch. Avec un nombre limité d'époques (10) et un entraînement principalement réalisé sur CPU, Adam permet d'obtenir de meilleures performances et une meilleure stabilité.

Le modèle ResNet18 utilisant le transfert d'apprentissage bénéficie des représentations déjà apprises sur ImageNet. Cette connaissance préalable permet une convergence plus rapide et de meilleures performances de généralisation que le CNN entraîné depuis zéro.

Les résultats confirment l'intérêt du transfert learning pour les jeux de données, où l'entraînement complet d'un modèle depuis zéro peut être plus difficile et plus coûteux en calcul.

Cependant, des résultats issue de la matrix de confusion, les erreurs de classification restent limitées avec 191 chats confondus avec des chiens et 138 chiens confondus avec des chats. Ces erreurs sont principalement dues à des similitudes visuelles entre les deux classes ou à des conditions d'acquisition difficiles (flou, faible contraste, posture inhabituelle). Malgré ces quelques confusions, le modèle ResNet18 obtient de bonnes performances globales grâce au transfert d'apprentissage.
---

## Limites et pistes d'amélioration
Comme limite:
* l'atteinte de la limite GPU lors des essais de la réalisation du devoir,
* Le nombre d'époques a été limité à 10 afin de maintenir des temps d'exécution raisonnables tout en permettant la comparaison des différentes approches 
* Les images ont été redimensionnées à **96 × 96 pixels** afin de réduire le coût computationnel de l'entraînement tout en conservant suffisamment d'informations visuelles pour distinguer les chats des chiens.
* Un **batch size de 16** afin de trouver un compromis entre la vitesse d'entraînement ; la consommation mémoire et la stabilité de l'optimisation.

Comme piste d'amélioration à notre travail, nous avons :
* Utilisation d'un GPU pour accélérer l'entraînement.
* Création d'un véritable ensemble de validation distinct du jeu de test.
* Augmentation du nombre d'époques.
* Exploration d'autres architectures pré-entraînées (MobileNetV2, EfficientNet).
* Analyse plus approfondie des erreurs de classification.
* Ajout de TensorBoard ou Weights & Biases pour le suivi expérimental.
