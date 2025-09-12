from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Vos questions et réponses Valorant
questions = [
    "Quelle agent est français ?",
    "A partir de combien de round la defense passe en attack et inversement ?",
    "C'est un FPS ?",
    "Combien d'equipe d'esport y'a t'il sur ce jeu dans le monde ?",
    "Quel est la meilleure equipe esport valorant ?",
    "MVP qu'est que sait ?",
    "Quel est le rank le plus haut ?",
    "Quel est le rank le plus bas ?",
    "Comment bien jouer a valo et progresser rapidement ?",
    "C'est quoi la spike ?"
]

answers = [
    "L'agent français est Fade.",
    "Après 12 rounds, les équipes changent de côté.",
    "Oui, Valorant est un FPS (jeu de tir à la première personne).",
    "Il y a plusieurs centaines d'équipes esport à travers le monde.",
    "Les meilleures équipes en 2025 sont G2 Esports, Paper Rex, Fnatic, Sentinels, Team Liquid.",
    "MVP signifie Most Valuable Player, le joueur le plus performant dans une partie ou un tournoi.",
    "Le rang le plus haut est Radiant.",
    "Le rang le plus bas est Fer.",
    "Pour progresser, il faut s'entraîner régulièrement, travailler sa visée, la communication et la connaissance des cartes.",
    "La spike est l'objet que l'équipe attaquante doit poser pour remporter une manche."
]

#Transformer les questions en vecteur TF-IDF il permet de savoir de combien de fois un mot apparaît dans la phrase et de à quel point ce mot est rare dans toutes les phrases
vectorizer = TfidfVectorizer().fit(questions)
questions_vectors = vectorizer.transform(questions)

def find_best_answer(user_question):
    user_vector = vectorizer.transform([user_question]) #Vectorise la question de l'utilisateur
    similarities = (questions_vectors * user_vector.T).toarray()
    