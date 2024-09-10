# Rapport

## 10 septembre

Partie 1:
    Parmit les différents tensor nous avons :

        - poids(w) : [784, 10] => 784 est le nombre de pixel de l'image que l'on possède (nombre d'entrée du percetron) et 10 est le nombre de sortie.

        - biais(b) : [1, 10] => 1 chiffre qui sera le biais ajouté lors du calcul de la sortie du neuronne et 10 est le nombre de sortie.

        - data_test : [7000, 784] => 7000 le nombre d'image total de test et 784 le nombre de pixel de chaque image.

        - data_train : [63000, 784] => 63000 le nombre d'image total d'entrainement et 784 le nombre de pixel de chaque image.

        - label_test : [7000, 10] => 7000 le nombre d'image total de test et 10 est le nombre de sortie.

        - label_train : [63000, 10] => 63000 le nombre d'image total de test et 10 est le nombre de sortie.

        - x : [5, 784] => 5 car c'est la taille du batch size et 784 est  le nombre de pixel de l'image.

        - y : [5, 10] => on fait le produit scalaires de deux matrices (x et w) on se retrouve donc avec 1 sorties par neuronne. Nous avons 10 neuronnes et nous prenons 5 images donc 5x10. 

        - t : [5, 10] => Pour les 5 images on regarde le label en 10 sorties.

        - grad : [5, 10] => On soustrait deux matrices de taille 5x10, on se retrouve donc avec une matrice de taille 5x10.

        partie test :

        - x : [1, 784] =>  1 car dans la partie test on veut voir une image et 784 est le nombre de pixel de l'image.

        - y : [1, 10] => on fait le produit scalaires de deux matrices (x et w) on se retrouve donc avec 1 sorties par neuronne. Nous avons 10 neuronnes et nous prenons 1 image donc 1x10. 

        - t : [1, 10] => Pour l'image on regarde le label en 10 sorties.

        - acc : [1] => On compte le nombre de bonne réponse.


