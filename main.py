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
#Pour qu’une IA puisse analyser, comparer, et traiter des questions, il faut convertir ces textes en représentations numériques (des vecteurs).
vectorizer = TfidfVectorizer().fit(questions)
questions_vectors = vectorizer.transform(questions)

def find_best_answer(user_question):
    user_vector = vectorizer.transform([user_question]) #Vectorise la question de l'utilisateur
    similarities = (questions_vectors * user_vector.T).toarray() #On multiplie les poids des mots pour voir dans quelle mesure les mots importants d’une question se retrouvent dans l’autre.
    #Plus le résultat est grand, plus les questions sont similaires (le minimum est 0 et le max est 1).
    best_idx = np.argmax(similarities) # retourne l'indice de la valeur maximale dans ce tableau
    score = similarities[best_idx][0] # recupere le score de l'indice de la valeur maximale
    if score == 0 : # Si le score est 0, cela signifie que la question utilisateur ne correspond à aucune question dans notre base de données
        return "Désolé, je ne connais pas la réponse à cette question." 
    # Sinon, on retourne la réponse associée à la question la plus proche de celle posée par l'utilisateur
    return answers[best_idx]

if __name__ == "__main__":
    print("Posez une question sur Valorant (tapez 'quit' pour arrêter) :")
    user_input = input("> ")
    print(find_best_answer(user_input))