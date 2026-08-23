# France Messagerie RAG Demo

Démonstrateur local d’interrogation des rapports France Messagerie
« Suivi du réseau des marchands de presse ».

Le projet part d’une contrainte volontaire : les fichiers PDF sont
considérés comme l’unique source disponible. Aucun fichier Excel,
PowerPoint ou autre fichier natif n’est requis.

## Objectifs du POC

Le POC doit notamment traiter les questions suivantes :

1. Identifier 5 à 10 faits marquants sur le réseau depuis décembre
   2021 jusqu’à juin 2026.
2. Produire une synthèse de l’évolution des kiosques depuis décembre
   2021.
3. Générer un graphique comparatif pour les Enseignes Presse.

L’objectif est d’obtenir des réponses plus fiables et plus
spécialisées qu’un RAG textuel généraliste.

## Pourquoi un RAG classique ne suffit pas

Les rapports contiennent principalement :

- des tableaux ;
- des courbes ;
- des graphiques en barres ;
- des camemberts ;
- des indicateurs chiffrés ;
- peu de texte narratif continu.

PyMuPDF peut lire les valeurs présentes dans les PDF, mais
l’extraction textuelle brute ne conserve pas toujours les relations
entre :

- une valeur ;
- sa ligne ;
- sa colonne ;
- sa période ;
- son indicateur ;
- son entité métier.

Le projet combine donc deux approches :

- un socle RAG textuel pour la recherche documentaire ;
- une extraction géométrique structurée pour les données chiffrées.

## État actuel

Le premier cas d’usage structuré est opérationnel sur la page
« Répartition et évolution par type d’enseigne ».

Le pipeline permet de :

- détecter la page par son titre ;
- détecter une ligne à partir de son libellé métier ;
- regrouper les mots selon leurs coordonnées ;
- reconstruire les nombres fragmentés ;
- convertir les valeurs françaises ;
- conserver la provenance géométrique ;
- produire des mesures structurées ;
- vérifier les totaux additifs ;
- distinguer les erreurs et les avertissements d’arrondi ;
- exporter les données en JSON ;
- générer un graphique temporel.

Les six rapports produisent actuellement :

```text
6 PDF × 5 catégories × 7 indicateurs = 210 mesures