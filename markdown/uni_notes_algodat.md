---
title:
    Algorithmen und Datenstrukturen (CS210), SS21
---

## Random notes

Analyseregeln:

- Schleife: Sei O(g(n)) eine obere Schranke fuer die Kosten eines Schleifendurchlaufs und
sei O(f(n)) eine obere Schranke fuer die Anzahl der Schleifendurchlaeufe. Dann
ist T(n) = O(f(n) * g(n)) die Kosten fuer die Schleife.
- Sequenz: Seien S1 und S2 zwei konsekutive Schritte eines Algorithmus mit Kosten
T1(n) = O(f(n)) und T2(n) = O(g(n)). Dann gilt fuer die Laufzeit der beiden Schritte
T(n) = O(max(f(n), g(n))).
- bedingte Anweisung: Gegeben eine bedingte Anweisung if (B) S1 else S2, mit Aufwand
    - h(n) fuer B
    - f(n) fuer S1
    - g(n) fuer S2,
dann betraegt die Laufzeit T(n) = O(h(n) + max(f(n), g(n)))

Datentyp/-struktur:

- Spezifikationsebene: ein Datentyp besteht aus einer speziellen Sorte, der Signatur,
der Zuordnung von Wertebereichen zu den Sorten und der Spezifikation der Operationen
- Algorithmische Ebene: die zu dem Datentyp zugehoerige Datenstruktur legt die
Repraesentation der Objekte fest. Weiterhin wird fuer jede Operation des Datentyps
ein Algorithmus angegeben.
- Programmierungsebene: Umsetzung in konkreter Programmiersprache wie Java.

Amortisierte Analyse: man berechnet die Laufzeit fuer eine Folge von gleichen
Operationen und teilt dann durch die Anzahl der Operationen -> amortisierte Kosten.

## Arrays

Motivation: Verwaltung von fester Menge an Datenelementen.

- Linear Search
    - Tbest     = O(1)
    - Tworst    = O(n)
    - Tavg      = O(n)
- Binary Search
    - Tbest     = O(1)
    - Tworst    = O(logn)
- COLA: Einfuegen in sortiertes Array beschleunigen
    - Tworst    = O(logn)
    - Tamo      = O(1)

## Listen

Motivation: Listen treten im Leben sehr oft auf.

- LinkedList
    - Tempty    = O(1)
    - Tadd      = O(n)
    - Tremove   = O(n)
    - Tcontains = O(n)
    - Tget(i)   = O(i)
- Skip-List
    - Tcontains = O(logn)
    - Tinsert   = O(n) im worst-case (schlecht!)
- Random-Skip-List (probabilistische Datenstruktur), average-Kosten:
    - Tcontains = O(logn)
    - Tinsert   = O(logn)
- adaptive Listen: Idee, haeufig benutze Elemente nach vorne zu schieben, dafuer
sind approximative Strategien notwendig
    - Frequency Count (FC): jedes Element besitzt Zaehler, der Anzahl Zugriffe speichert
    und die Liste danach sortiert. Problem: Element entfernen und wieder hinzufuegen
    setzt Frequenz zurueck, Aufwand sehr hoch
    - Move-to-Front (MF): beim Zugriff auf Element wird es nach vorne geschoben,
    geringer Aufwand
    - Transpose (T): vertausche Element bei Zugriff mit Vorgaenger. Sehr langsame
    Adaption bei Aenderung der Zugriffswahrscheinlichkeiten, niedriger Aufwand
- Stack (push/pop/peek, LIFO)
    - Tpush     = O(1)
    - Tpop      = O(1)
    - Tpeek     = O(1)

## Hashverfahren

Motivation: wir haetten gerne eine effiziente Moeglichkeit, key-value-Pairs zu
speichern.

Open Hashing (Ueberlaeufer werden in overflow area gespeichert, a=Belegungsfaktor):

- direkte Verkettung (jeder Bucket ist LinkedList)
    - Tinsert   = O(1)
    - Tsearch   = O(1) (best)
    - Tsearch   = O(n) (worst)
    - Tsearch   = O(a) (average), daher ist kleines a zu bevorzugen
    - durchschn. Listenlaenge ist Binomialverteilt
- separate Verkettung (statt LinkedList wird erster Datensatz und Pointer auf LinkedList gespeichert)
    - sorgt letztendlich nur fuer niedrigeren Speicherplatzbedarf

Closed Hashing (Ueberlaeufer werden in Hashtabelle gespeichert):

- lineares Sondieren (Sondierungsfolge 1,2,3,4,...)
    - Tsearch   ~ 1/(1-a) -> fuer a~1 sehr schlecht weil Clustering
- quadratisches Sondieren (Sondierungsfolge 1,-1,2,-2,4,-4,...)
    - Tsearch ist effizienter als bei linearem Sondieren, aber sekundaere Clusterung
    ist auch hier ein Problem, da Sondierungsfolge gleich
- zufaelliges Sondieren (Double-Hashing)
    - entweder zufaellig oder unter Verwendung von zwei unabhaengigen Hashfunktionen
    - ziemlich effizient, aber nicht genau diskutiert
- hybrides Hashing (Kombination von Verkettung und Sondierung)
    - extra Ueberlaufliste, bei Kollision wird Datensatz an Ende der Ueberlaufkette angehaengt
    - leicht effizienter als Verfahren mit Sondierung und Verkettung

Dynamische Hashverfahren (kontinuierliche Reorganisation bei Wachsen/Schrumpfen):
- globale Reorganisation: bei zu hohem Belegungsfaktor wird Tabelle vergroessert und alte
Datensaetze reinkopiert
    - amortisiert O(1); die Einfuegeoperation, die Reorganisation ausloest, hat O(n)
    - Hashtabelle ist waehrenddessen "offline"
- kontinuierliche Reorganisation: staendige Anpassung der Hashtabelle

- lineares Hashing: bei zu hohem Belegungsfaktor werden neue Buckets am Ende eingefuegt
und vorhandene Buckets der Reihe nach entzweit (zwei Hashfunktionen notwendig mit Spalt-Eigenschaft)
    - Tworst    = O(n) (fuer Suchen, Einfuegen und Loeschen)
    - Tavg      = O(1) (auch best-case)
    - Leistung entspricht im Prinzip der Verkettung mit der Garantie, dass der
    Belegungsfaktor eine Grenze nicht ueberschreitet

Worst-Case optimierte Verfahren:

- Second-Choice Hashing: zwei Hashtabellen mit unterschiedlichen Hashfunktionen,
Datensatz wird in niedrig belegtere Tabelle eingefuegt
    - bessere worst-case Kosten als bei Verkettung, dazu noch asymptotischer Durchschnitt
- Robin-Hood Hashing (Variante von linearem Sondieren): wird bei Einfuegen von x ein
Datensatz y gefunden, der naeher an Heimatadresse ist als x, wird x bei Platz von y
abgespeichert und Sondierung mit y fortgesetzt
    - Varianz der Suchkosten wird reduziert
- Kuckucks-Hashing (Kombination von Second-Choice- und Robin-Hood-Verfahren)
    - Tworst    = O(1) fuer Suchen und Loeschen
    - Tworst    = O(1) amortisiert fuer Einfuegen

## Trees

Motivation: Repraesentation von Hierarchien.

- Heap (implementiert Priority Queue)
    - Tpeek     = O(1)
    - Tremove   = O(logn) (worst-case)
    - Tinsert   = O(logn) (worst-case)
- Binary Search Tree (langsamer als Hashmaps, unterstuetzen aber mehr Operationen
wie z.B. nexthighkey)
    - Tworst    = O(h) (mit h als Baumhoehe. Es gilt hmax=n, falls Baum zu Liste
    entartet ist, und hmin=log(n+1) fuer vollstaendigen Baum)
- AVL-Trees (Problem der Baumentartung durch Reorganisation loesen)
    - Hoehe ist logarithmisch beschraenkt
    - Tinsert   = O(logn)
- 2-3-4-Trees <=> RB-Trees
    - Garantie, dass alle Blaetter gleiche Hoehe haben
    - Transformation in RB-Baum mit h = O(logn) moeglich

## Graphs

Motivation: Menge von Objekten mit Beziehung zueinander (Transportnetze, Soziale
Netzwerke...).

- Topsort (beschraenkt durch Finden von Knoten mit indeg=0)
    - count-array:  T = O(m) + O(n^2)
    - queue:        T = O(m) + O(n) (fuer kleine m mit m=O(n) gilt also T=O(n))
- Traversing (breadth- and depth)
    - T         = O(n+m)
- starke Zusammenhangskomponenten
    - T         = O(m+n)
- Dijkstra (shortest-path tree)
    - T         = O(n^2) (Gelb wird verwaltet durch Nichtvorhandensein in Gruen)
    - T         = O(m logn) (Verwaltung durch Heap)
    - T         = O(m + n logn) (durch Fibonacci-Heaps)
- Prim (minimal spannender Baum)
    - n Iterationen, pro Iteration wird ein Element aus Heap entfernt und
    insg. maximal m Nachbarknoten in Heap eingefuegt
    - T         = O(m logn)
- Kruskal (minimal spannender Baum)
    - gleichzeitiger Aufbau von mehreren Untergraphen durch Wahl der Kanten mit
    geringsten Kosten (durch Heap realisierbar)
    - Verschmelzen der Untergraphen und Verwerfen von Kanten zwischen Knoten,
    die im selben Untergraphen liegen

## Sorting

Motivation: Datensaetze sortieren und Zugriffszeiten minimieren.

- Selection-Sort (fuer jede Position das richtige Element auswaehlen)
    - T         = O(n^2)
- Insertion-Sort (fuer jedes Element die richtige Position auswaehlen)
    - Tbest     = O(n - 1)
    - Tworst    = O(n^2)
- Radix-Sort (keine Ordnungsrelation notwendig; Sortieren durch Partitionierung)
    - T         = O(L * (m+n)) (O(n) fuer L konstant und m < n, wow!)
- Merge-Sort (billiges divide, teures merge), asymptotisch optimal
    - Tbest     = O(n logn)
    - Tworst    = O(n logn)
    - Tavg      = O(n logn)
- Quick-Sort (teures divide, kein merge)
    - Tbest     = O(log n)
    - Tworst    = O(n^2)
    - Wahl des Pivots: Clever Quicksort (Median des linken, rechten und mittleren
    Elements)
- Heap-Sort (n Elemente in Heap packen und nacheinander entnehmen)
    - Tworst    = O(n logn) (besser als Quick-Sort mit Bottom-Up Version)
