# Identification possibles exclusions

# TODO: reminder
-> faudra que nos exclusions on pense aux virgules etc.
Cf dans ce que j’ai renvoyé dans les contextes elles sont supprimées
(Pas dans le nettoyage à proprement parler mais dans la tokenisation que je fais à l’arrache)
i.e. :  motif \b[\w-]+\b -> garde surtout lettres/chiffres/_/tiret, pas la virgule

## CRS/POLICE

- des compagnies républicaines de sécurité,10
- les compagnies républicaines de sécurité,8
- aux compagnies républicaines de sécurité,3
- une compagnie républicaine de sécurité,3
- deux compagnies républicaines de sécurité,2
- CRS compagnies républicaines de sécurité,2
- quatre compagnies républicaines de sécurité,2
- nationale républicaine de,7
- compagnie républicaine de,5
- police nationale républicaine de proximité,7 -> plus touchy ??
- la Garde républicaine de la,1
- Garde républicaine,1
- Garde républicaine entre,1
- gardes républicains Nous,1
- police républicaine et,9 -> plus touchy, du sens de la qualifier comme ça dans les débats

## lois ?

- respect des principes de la République et de lutte contre le -> loi séparatisme sous un autre nom
- nouvelle organisation territoriale de la République et MAPTAM de modernisation de
- de la République dite loi,66
- de la République loi NOTRe,10
- de la République dite NOTRe,9
- de la République NOTRe à,1
- intégration républicaine CIR,5
- d intégration républicain le CIR,1
- d intégration républicaine le CIR,1
- de la République en Nouvelle-Calédonie,8 -> un nom de loi ?
- de République écologique,3 -> loi ?
- de la République en Polynésie,6 -> risque nom de loi ?
- de la République dite séparatisme,2
- pour une République numérique
- d intégration républicaine à Mayotte,2
- de la République et MAPTAM,2
- reconquête républicaine s,1
- reconquête républicaine concernent,1
- reconquête républicaine Des,1
- quartier(s) reconquête républicaine -> environ 90 total
- contrat d'intégration républicaine -> 40, sans doute plus avec variations
- Haut-commissaire/haut-commissariat de la République en Nouvelle-Calédonie -> 8 ?.
- contrat d'engagement républicain -> 415, mais pas mal qui restent (passe de 11429 à 11371 interv)
- décider dans ce cas pour forme républicaine du gouvernement (art 89 constitution)

## Groupes
en vrai : Communiste, républicain, citoyen et écologiste
->  -> aviser avec et sans virgules selon nettoyage texte + feintes dans la graphie utilisée
- groupe communiste républicain citoyen et,8
- groupe Communiste républicain citoyen et,6
- CRCE communiste républicain citoyen et,1
- Communiste républicain citoyen,6
- groupe socialiste républicain et citoyen,3
- Parti républicain américain,1
- groupeLes Républicains ne,1 -> soucis espace et donc regex \b exclu pas ?
- Grâce aux Républicains,5 -> plus d'ex républicains en discussion
- députés Républicains n,1
- Macronistes Républicains lepénistes,1
- députés Républicains il,1
- élus Républicains nous,1 -> ?
- hémicycle Républicains au,1 ???
- Républicains ex-Républicains et,1 ??


## Pays ?

- et la République d Irlande,7
- Irlande républicaine une,1
- de la République d Artsakh,6
- Union des républiques socialistes soviétiques,4
- de la République démocratique allemande,2
- de la République d Irlande,2
- de la République de Macédoine,2
- à la République socialiste soviétique,1
- aussi la République arabe sahraouie,1
- de la République romaine Nihil,1
- grave république romaine,1
- la République romaine,4
- dans la république de Haïti,1
- des républiques socialistes,4
- les ex-républiques soviétiques,1
- l ex-République yougoslave,1
- la République ukrainienne,1 (parfois des maj qui merdent ? ou juste on avait pas ?)
- la République sahraouie,1
- la République bolivarienne,1
- gouvernement républicain afghan,1
- La République girondine,1 ? #lesgigi
- votre république bananière,1 # débat possible : imp mais on a viré soviétique 

## autres ?

- L Est républicain non,1
- L Est républicain Quand on,1
- le Républicain n,1
- otages républicains espagnols,2
- axe République-Bastille je,1 (géo parisienne ? ahaha, mais apparait qu'une fois)

## Discussion et remarques

- est-ce que occurrences Ve République sont genre très loi et rappel de loi ?  -> mais compliqué à virer tellement c'est large en vrai
- par contre : la forme républicaine du gouvernement revient souvent et semble être une forme de rappel à la la loi ou je ne sait pas quoi (article 89 constitution ?)
- blablabla république dite blalabla (dans des formes différentes selon les virgules et tout, pour identifier les noms de loi ?)
- pour une République numérique
- d engagement républicain (un nom de loi ? pacte d'engagement blabla bla ? truc associatif ? aviser pour pas virer si important) Et surtout, le fait même qu'il y ait république dans le nom est un signal en soit. Ça se justifie de dire : non, on a viré l'essentiel problématique, mais si les lois contiennent républicain, ça a du sens aussi.
- cf pacte/engagement/reconquête républicain/e -> pour moi faut garder, mais si spécifique loi on peut en causer.
- reconquête républicaine y a peut être une loi ?
- pour reconquête machin peut-être **quartier reconquête républicaine** ?
- République sociale revient, vérif si loi ? -> vu, non c'est ok.
- de la République et MAPTAM
- **aux Républicains,12** (peut-être parfois sans majuscule ?) / les bancs républicains de cet,2 / les parlementaires Républicains qui y,1 / parlementaires Républicains qui,1 / le sénateur républicain 106 millions,1 / collègues Républicains parce,2 / députés Républicains n,1 -> AVISER POUR GROUPE LES RÉPUBLICAINS ? MAIS RISQUE DE SUPPR DES TRUCS VALIDES
- L Est républicain non,1 / L Est républicain Quand on,1 -> journal ?
- j'ai un Président la République sur la,1 qui traîne, mais sans doute car le texte est flagué pour un autre truc valable
- socialiste et républicain ? (groupe ?)
- votre république bananière,1 # débat possible : pour moi c'est imp, mais on a viré soviétique

## Pour l'instant

- j'ai vérifié le contexte 1 jusqu'à 6230/6230 (n<1 à partir de 1622) = DONE
- j'ai vérifié le contexte 2 jusqu'à 3000 (n<1 à partir de 1565)
- j'ai vérifié le contexte 5 jusqu'à je sais plus
