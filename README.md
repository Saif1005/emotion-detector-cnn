# 🎭 Détection d'Émotions en Temps Réel avec CNN et YOLO

Ce projet utilise un modèle CNN personnalisé pour détecter les émotions humaines à partir des visages détectés dans une vidéo en temps réel (via webcam). Il utilise YOLO pour détecter les visages, puis un modèle de classification CNN pour prédire l’émotion. En cas d’émotion particulière détectée, une alerte peut être envoyée par email.

## 📦 Fonctionnalités

- Détection de visages avec **YOLOv8**
- Classification des émotions via un **modèle CNN entraîné**
- Prédiction en **temps réel** depuis une webcam
- **Envoi d'email automatique** lorsqu'une émotion spécifique est détectée (ex : peur, colère)

## 🧠 Émotions reconnues

Le modèle peut détecter les émotions suivantes :
- 😠 Colère
- 🤢 Dégoût
- 😨 Peur
- 😀 Heureux
- 😢 Triste
- 😲 Surpris
- 😐 Neutre

## 🛠️ Installation

### 1. Cloner le projet
```bash
git clone  https://github.com/Saif1005/emotion-detector-cnn.git
cd emotion-detector-cnn
