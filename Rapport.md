# Rapport
Arthur DESBIAUX p2006393
Valentin CUZIN RAMBAUD p2003442
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
## 12 septembre
Partie 2 :
    Détail méthodologie : On a écrit une classe ShallowNet étendu de nn.Module que l'on initialise avec une couche cachée contenant $N$ neurones, et une couche de sortie contenant 10 neurones de sorties (par rapport au label de sortie codé sur 10). Dans la méthodes forward, on implémente la logique du réseau, à savoir : couche caché => fonction d'activation relu => couche de sortie. le neurone fait une somme pondéré de ses entrées. Le poid de chaque entré est initialisé aléatoirement. On charge la données en découpant en 3 parties: train, validation puis test. On entraine le modèle de façon itérative sur un échantillon $X$ mini-batch de train.A la fin d'une epoch on **retient le dernier modèle** avec son erreur calculer sur le jeu de validation. Cela nous permet de retenir des modèles qui varient en fonction des hyper-paramètres suivant : taille du batch, nombre d'epoch, nombre de neurones pour la couche cachée, taux d'apprentissage.
    Pour finir on retient le meilleur modèle pour l'envoyer au test, et mesurer le taux de réussite de sa prédiction.

Meilleur paramètre pour le moment : 
Batch Size,Hidden Num,Learning Rate,Epochs
3, 300, 0.01, 20
avec un score de 0.9807
    
Par la suite:
1. Faut voir TensorBoard, afin de comprendre l'influance des divers hyper-paramètres.
2. L'arrêt précoce pour enlever l'hyper-paramètre du nombre d'epoch.
3. La bibliothèque ray Tune, pour cette argument suivant : " includes the latest hyperparameter search algorithms, integrates with TensorBoard and other analysis libraries, and natively supports distributed training".
4. formater le code pour aller plus vite sur la partie 3 et 4.

