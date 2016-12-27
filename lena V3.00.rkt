#lang racket

(require math/statistics)
  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; PARTIE RESEAU NEURONAL de Christopher Wellons - skeeto - ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Scheme Artificial Neural Network
;; TODO
;; - learning rate
;; - load/save weights

;;; Supporting functions

;; Stack push macro
(define-syntax push!
  (syntax-rules ()
    ((push item place)
     (set! place (cons item place)))))

;; Sum weights in given list
(define (sum-weight lst)
  (if (empty? lst) 0
      (+ (* ((caar lst)) (cadar lst))
         (sum-weight (cdr lst)))))

;; Sigmoid learning function
(define (sigmoid x)
  (/ (+ 1.0 (exp (- x)))))

;; Generate a new random weight
(define (rand-weight)
  (- (random) 0.5))

;; Generate a new random threshold
(define (rand-theta)
  (- (* (random) 4) 2))

;;; Neuron functions

;; Train neurons in weight list
(define (train lst err)
  (if (empty? lst)
      '()
      (let ((n (caar lst))
            (w (cadar lst)))
        (n 'sum (* err w))
        (cons (list n (+ w (* (n) err)))
              (train (cdr lst) err)))))

;; Create a new neuron
(define (new-neuron)
  (let ((theta (rand-theta))
        (backward '())
        (cache #f)
        (trained #f)
        (train-sum 0))
    ;; Neuron function with closure
    (lambda ([method 'activate] [arg '()])
      (cond
       ((eq? method 'backward)
        (push! (list arg (rand-weight)) backward))
       ((eq? method 'set)
        (set! cache arg))
       ((eq? method 'reset)
        (set! cache #f)
        (set! trained #f)
        (set! train-sum 0))
       ((eq? method 'sum)
        (set! train-sum (+ train-sum arg)))
       ((eq? method 'list)
        (map (lambda (el) (cadr el)) backward))
       ((eq? method 'train)
        (if (not trained)
            (set! backward (train backward
                                  (* cache
                                     (- 1 cache)
                                     train-sum)))
            (set! trained #t)))
       ((eq? method 'activate)
        (if cache
            cache
            (begin
              (set! cache (sigmoid (sum-weight backward)))
              cache)))))))

;;; Layer functions

;; Create a new neuron layer
(define (new-layer n)
  (if (= n 0) '()
      (cons (new-neuron) (new-layer (- n 1)))))

;; Link two layers together
(define (link-layers left right)
  (if (or (empty? left) (empty? right))
      '()
      (begin
        ((car right) 'backward (car left))
        (link-layers (cdr left) right)
        (link-layers left (cdr right)))))

;; Link up layers in an unlinked ann
(define (link-ann ann)
  (if (= (length ann) 1) '()
      (begin
        (link-layers (car ann) (cadr ann))
        (link-ann (cdr ann)))))

;; Hard set a layer of neurons
(define (set-layer layer in)
  (if (empty? layer) '()
      (begin
        ((car layer) 'set (car in))
        (set-layer (cdr layer) (cdr in)))))

;; Activate a layer, which activates all layers behind it
(define (run-layer layer)
  (if (empty? layer) '()
      (cons ((car layer)) (run-layer (cdr layer)))))

;; Reset a single layer
(define (reset-layer layer)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'reset)
        (reset-layer (cdr layer)))))

;; Train a layer, which back-propagates
(define (sum-layer layer out desired a)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'sum (* a (- (car desired) (car out))))
        (cons (car out)
              (sum-layer (cdr layer)
                           (cdr out)
                           (cdr desired)
                           a)))))

;; Run 'train on each neuron in layer
(define (train-layer layer)
  (if (empty? layer)
      '()
      (begin
        ((car layer) 'train))))

;; Run training on all layers from front (pass in reversed)
(define (train-layers rev-ann)
  (if (empty? rev-ann)
      '()
      (begin
        (train-layer (car rev-ann))
        (train-layers (cdr rev-ann)))))

;;; ANN functions

;; Create new ann based on specification
(define (new-ann spec)
  (let ((ann (map new-layer spec)))
    (link-ann ann)
    ann))

;; Reset each neuron in ann
(define (reset-ann ann)
  (if (empty? ann)
      '()
      (begin
        (reset-layer (car ann))
        (reset-ann (cdr ann)))))

;; Get output of ann
(define (run-ann ann in)
  (set-layer (car ann) in)
  (let ((out (run-layer (last ann))))
    (reset-ann ann)
    out))

;; Train the ann
(define (train-ann ann in desired [a 1])
  (set-layer (car ann) in)
  (let ((out (run-layer (last ann))))
    (sum-layer (last ann) out desired a)
    (train-layers (reverse ann))
    (reset-ann ann)
    out))

;;; App functions

;; round to binary 1's and 0's
(define (round-output out)
  (map (compose inexact->exact round) out))

;; Turn integer into list suitable for an ANN
(define (int->blist int size)
  (if (= 0 size)
      '()
      (cons (remainder int 2)
            (int->blist (floor (/ int 2)) (- size 1)))))

;; Turn ANN output into integer
(define (blist->int lst)
  (if (null? lst)
      0
      (+ (car lst) (* 2 (blist->int (cdr lst))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;               PARTIE LECTURE FICHIER CSV                 ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;; Reading a table from a port where it resides in CSV format.
;; Copyright (C) 2003 Wolfgang Jährling <address@hidden>
;;
;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 2 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

(define field-delimiter #\;)

;; Return a procedure that calls CONSUMER with three arguments: The
;; value returned by the PRODUCER applied to the procedures arguments,
;; a list that is initially empty, and a thunk to restart this process
;; with the value given by the PRODUCER added at the beginning of the
;; list given to the CONSUMER.
(define (collectrec producer consumer)
  (lambda args
    (letrec ((loop (lambda (lst)
                     (let ((x (apply producer args)))
                       (consumer x lst (lambda ()
                                         (loop (cons x lst))))))))
      (loop '()))))

;; Read and return a field, that ends with the configured delimiter
;; character, or return false at the end of a line, or the eof-object
;; at end of file.
(define read-field
  (collectrec read-char
              (lambda (c chars loop)
                (cond ((eof-object? c) c)
                      ((char=? c field-delimiter)
                       (apply string (reverse chars)))
                      ((char=? c #\newline) #f)
                      (else (loop))))))

;; Read a line and split it up into a list of fields which gets
;; returned, or false at the end of the file.
(define read-row
  (collectrec read-field
              (lambda (f fields loop)
                (cond ((not f) (reverse fields))
                      ((eof-object? f) #f)
                      (else (loop))))))

;; Read a table and return it as a list of rows, each row being a list
;; of fields, which are strings.
(define read-table
  (collectrec read-row
              (lambda (r rows loop)
                (if (not r)
                    (reverse rows)
                  (loop)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                   Luke Miles, June 2015                    ;;;
;;                                                              ;;
;                                                                ;
;;                            K-means                           ;;
;;;                                                            ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;(require (only-in unstable/list group-by))

;; splits a list ls into k non-empty & disjoint sublists
(define (split-into ls k)
  (define size (quotient (length ls) k))
  (let R ([ls ls] [k k])
    (if (= k 1)
      (list ls)
      (let-values ([(soon later) (split-at ls size)])
        (cons soon (R later (sub1 k)))))))

;; calculates the mean point in a list of points
(define (point-mean points)
  (define length@points (length points))
  (map (λ (ls) (/ (apply + ls) length@points))
       (apply map list points)))
         ;(zip points)))

;; squared euclidean distance
(define (distance p1 p2)
  (for/sum ([x1 (in-list p1)]
            [x2 (in-list p2)])
    (expt (- x2 x1) 2)))

;; the closest center to point in centers
(define (closest centers point)
  (argmin (curry distance point) centers))

;; returns the value x such that (f x) = x
(define (fixed-point f start [same? equal?])
  (let R ([x start])
    (let ([f@x (f x)])
      (if (same? x f@x)
        x
        (R f@x)))))

;; given a list of points and centers,
;; assign each point to the nearest center,
;; then return, for each center, the mean of the points closest to it
(define (make-next-centers points centers)
  (map point-mean (group-by (curry closest centers) points)))

;; divides the set S of points into k cluster
(define (cluster points k)
  (define first-centers (map point-mean (split-into points k)))
  (define final-centers
    (fixed-point (λ (centers) (make-next-centers points centers))
                 first-centers))
  (values final-centers (group-by (curry closest final-centers) points)))

;TODO? put a contract on cluster
(provide cluster)


;utilisation par 

;(define-values (centers clusters) (cluster points 5))

;(printf "centers are about: ~a\n" (map (curry map exact-round) centers))

; ou 5 est le nombre de cluster anticipé

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                  PARTIE         PERSO                    ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;centrage de la moyenne et reduction de la variance
(define (centrage-reduction-liste lst)
  (map (lambda (x)
         (cond
           [(number? x)(/ (- x (mean lst)) (stddev lst))]
           [else "indeterminé"])
        )
       lst))

;iota (n x s) : signifie : une liste de n nombres commençant à x et espacés de s. Par défaut x = 0 et s = 1.
(define ($iota n x e)
(if(= n 0)
   '()
   (cons x ($iota (- n 1)(+ x e)e))))

(define (extract-col colonne liste-de-liste) ; colonne est la position de l'ement dans la sous-liste en comptant de zero
  (map(lambda(x)
     (cond
       [(null? x) '()]
       [else (string->number(list*  (list-ref x (- colonne 1)) ))]
       ))   liste-de-liste)
)

;generer la moyenne mobile n sur colonne2

;(define (MMx n list)            ; moyenne mobile x sur la liste
;  (cond
;    [(< (length list) n) '()]
;    [else (append (transforme-en-liste(average(extraction-n-element n list)))
;                  (MMx n (rest list)))]
;    ))

(define (MMx n list)
   (cond
    [(< (length list) n) '()]
    [else (append (transforme-en-liste(mean(extraction-n-element n list)))
                  (MMx n (rest list)))]
    ))


(define (average the-list)                ;probleme de la division par zero !!!
  (let loop ((count 0.000000000000000000001) (sum 0) (args the-list))
 (if (not (null? args))
     (loop (add1 count) (+ sum (car args)) (cdr args));then expression
     (/ sum count);else expression
     )))
(define (extraction-n-element n liste)   ; extrait n elements de la tete de la liste 
  (reverse (list-tail(reverse liste)     ;et restitue la liste extraite
                     (- (length liste) n))))

(define (transforme-en-liste element) ;transformer un element en liste
  (cons element '()))

(define (egaliseur-de-liste longueur liste) ;rallonger la liste à la longueur
  (cond
    [(= (length liste) longueur) (append (liste '()))]  ; ne rien faire car la longueur de la liste convient
    [(> (length liste) longueur)(list-tail liste (- (length liste) longueur ) )]     ; effacer n elements en tete de liste ( le filtrage induit un retard)
    [else ( append
            (make-list (- longueur (length liste)) 'indetermine)
            liste)]
    ))

(define (lag x liste)
  (cond
    [(= x 0) '()]   ; ne rien faire arg vaut 0 donc pas de retard !
    [(> x 0)                      ; x positif donc on pousse la liste vers le futur , on tronque en fin de liste et on rajoute des eements en debut à la date t on a la valeur de t-1
      (cond                        
    [(= (length liste) 0) '()]
    [else(append  (make-list (abs x) 'indeterminé) (reverse(list-tail (reverse liste)  x)) ) ]
   )
     ]
    [(< x 0)                      ; x negatif, on remonte la liste vers le passé on a donc a la date t la valeur de t+1, on trnque en debut de liste et on rajoute des elements en fin de liste
      (cond                        
        [(= (length liste) 0) '()]
        [else(append (list-tail liste (abs x)) (make-list (abs x) 'indeterminé)) ]
   )]))

(define (-* arg1 arg2)     ;operation de soustraction renvoyant indeterminé si les arguments ne sont pas numeriques
  (cond
    [(and (number? arg1) (number? arg2)) (- arg1 arg2)]
    [else 'indetermine]
    ))

(define (test-longueur-listes arg1 arg2 arg3 arg4 arg5 arg6 arg7)
  (cond
    [(and (length arg1)
          (length arg2)
          (length arg3)
          (length arg4)
          (length arg5)
          (length arg6)
          (length arg7)
          )
     'longueur-OK]
    [else 'probleme-longueur-de-liste]
    ))

(define (strategie-alpha-fonction1 arg1 arg2 arg3 arg4 arg5 arg6 arg7) ; les arguments se referrent aux entrees correspondantes
  (cond    ; les arguments sont dans cet ordre colonne2 cloture-lag1 cloture-variation-1 MM20 MM50 MM20-lag1 MM50-lag1
    [(and (number? arg1)(number? arg2)
             (number? arg3)(number? arg4)(number? arg5))
     (cond     ; ici on est dans le cas ou c'est determiné, validation à determiner
     [(and(> (- arg4 arg5) 0)
         (> (* (- arg4 arg5) (- arg6 arg7 )) 0 )
         (< (- arg2 arg1) 0)
         ) 'valide]
     [else 'non-valide])
     ]
    [else 'indeterminé]
    ))
; ARG 3 NON UILISE, A VOIR

(define (strategie-alpha-fonction arg1 arg2 arg3 arg4 arg5 arg6 arg7) ; les arguments se referrent aux entrees correspondantes
  (cond    ; les arguments sont dans cet ordre colonne2 cloture-lag1 cloture-variation-1 MM20 MM50 MM20-lag1 MM50-lag1
    [(and (number? arg1)(number? arg2)
             (number? arg3)(number? arg4)(number? arg5))
     (cond     ; ici on est dans le cas ou c'est determiné, validation à determiner
     [(or(< (- arg2 arg1) 0)
          (> (- arg2 arg1) 0)
         )'valide]
      [else 'non-valide])
     
     ]
    [else 'indeterminé]
    ))


(define (fonction-extractor arg1 arg2 arg3 arg4 arg5) ; fonction d'extraction du cours de cloture,
  ;du cours de mm20, mm50,et hausse, et validite de la strategie ( transformer en 1), arguments dans l'ordre
(cond
  [(equal? arg5 'valide)  (append (transforme-en-liste arg1)
                                  (transforme-en-liste arg2)
                                  (transforme-en-liste arg3)
                                  (transforme-en-liste arg4)
                                  (transforme-en-liste 1))]
  [else null]
  ))
(define (fonction-extractor1 arg1 arg2 arg3 arg4 arg5) ; fonction d'extraction du cours de cloture,
  ;du cours de mm20, mm50,et hausse, et validite de la strategie ( transformer en 1), arguments dans l'ordre
(cond
  [(equal? arg5 'valide)  (append (transforme-en-liste arg1); cours de cloture
                                  (transforme-en-liste arg2); mm20
                                  (transforme-en-liste arg3); mm50
                                  (transforme-en-liste arg4); hausse
                                  (transforme-en-liste 1)  ; c'est probablement ici qu'il faut modifier pour integrer les baisses
                                  )]
  [else null]
  ))






(define (removeAll s L) ;(filter (lambda (x) (not (equal? x '2))) ‘(1 2 1 2 3)) ↔ (removeAll ‘2 ‘(1 2 1 2 3))
(cond ((null? L) '())
((equal? (car L) s) (removeAll s (cdr L))) ; la différence c’est qu’on ne s’arrête pas, on doit chercher
(else (cons (car L) (removeAll s (cdr L))))))

(define (cumul liste)
      (cond
        [(= (length liste ) 0) 0]
        [else (+ (first liste)(cumul (rest liste)))]
          ))

;centrage sur la moyenne
(define (centrage-liste lst)
(map (lambda (x)
       (- x (mean lst)))
       lst))
;reduction de la variance
(define ( reduction-liste lst)
  (map (lambda (x)
         (/ x (stddev lst)))
       lst))

(define ( validateur arg1 arg2 arg3 arg4 arg5 arg6 arg7)
  (map (lambda(arg1 arg2 arg3 arg4 arg5 arg6 arg7)
          (cond    ; les arguments sont dans cet ordre colonne2 cloture-lag1 cloture-variation-1 MM20 MM50 MM20-lag1 MM50-lag1
    [(and (number? arg1)(number? arg2)
             (number? arg3)(number? arg4)(number? arg5))
     (cond     ; ici on est dans le cas ou c'est determiné, validation à determiner
     [(or(< (- arg2 arg1) 0)
          (> (- arg2 arg1) 0)
         )'valide]
      [else 'non-valide])
     
     ]
    [else 'indeterminé]
    )
         )
       arg1 ;colonne2
       arg2 ;cloture-lag1
       arg3 ;cloture-variation-1
       arg4 ;MM20
       arg5 ;MM50
       arg6 ;MM20-lag1
       arg7)) ; MM50-lag1

(define (tableur arg1 arg2 arg3 arg4 arg5)
  (map (lambda(arg1 arg2 arg3 arg4 arg5)
               (cond
  [(equal? arg5 'valide)  (append (transforme-en-liste arg1); cours de cloture
                                  (transforme-en-liste arg2); mm20
                                  (transforme-en-liste arg3); mm50
                                  (transforme-en-liste arg4); hausse
                                  (cond                                            ; 1 ou 0 pour la valeur de sortie du reseau 
                                    [(> arg4 0) (transforme-en-liste 0)]; baisse
                                    [else (transforme-en-liste 1)]; hausse;
                                 ))]
  [else null]
  ))
         arg1 ;colonne2
         arg2 ;MM20
         arg3 ;MM50
         arg4 ;cloture-variation-1
         arg5 ;strategie-alpha
         )
       )

(define (formateur liste-valide flag-validation)     
  (removeAll null (map (lambda (x)
       (cond
         [(= flag-validation (fifth x)) x    ]            
         [else '()]))
       liste-valide) )) 

(define (sensibilite vp fn)   ; Sensibilté d'un signe pour un diagnostic est la probabilité que le signe soit présent chez les individus atteints par la maladie recherchée.
  (/ vp (+ vp fn)))

(define (specificite vn fp)  ; Spécificité d'un signe pour un diagnostic est la probabilité que le signe soit absent chez les individus non atteints par la maladie recherchée.
  (/ vn (+ vn fp)))

(define (vpp vp fp)    ; valeur preditive positive
  (/ vp (+ vp fp)))

(define (vpn vn fn)   ;valeur predictive negative
  (/ vn (+ vn fn)))

(define (youden sensibilite specificite)  ; (sensibilité + spécificité - 1). " Indice négatif = test inefficace ; Indice se rapproche du 1 = test efficace "
       (- (+ sensibilite specificite) 1))

;partie tri de la liste de coordonnées selon les valeurs croissantes des ordonnées
(define (insert1 L M)   ; on recherche un classement coordonnées selon les y decroissants puis a x decroissants
  
   (if (null? L) M
		(if (null? M) L
			(cond
                          [(> (second(car L)) (second(car M)))                        ; etagement en y
                           (cond
                             [(= (first(car L)) (first(car M))) (cons (car M) (insert1 (cdr M) L))  ]        ;pas etagement en x ?
                             [else (cons (car M) (insert1 (cdr M) L))])]                                         ; etagement en x 
           
                          [(= (second(car L)) (second(car M)))                        ; pas d'etagement en y, y identique
                           (cond
                             [(< (first(car L)) (first(car M))) (cons (car M) (insert1 (cdr M) L))  ]
                             [else (cons (car L) (insert1 (cdr L) M))])]
                          [else (cons (car M) (insert1 (cdr M) L))]
                          
                             
                       ))))

(define (insertionsort2 L)
	(if (null? L) '()
		(insert1 (list (car L)) (insertionsort2 (cdr L)))))





                       ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
                       ;;;                                                                                              ;;;
;;;;;;;;;;;;;;;;;;;;;;;;                        PARTIE PROGRAMME RESEAU NEURONAL   - application BOURSE                   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
                       ;;;                                                                                              ;;;
                       ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; initialisation
(define col 5)
(define fichier-data "C:/Users/Administrateur/documents/PX1.csv")
(define longueur-base-training 40)  ;nbre d'items de hausse et de baisse ( longueur reelle de base formation = x2)
(define fichier-sortie "C:/Users/Administrateur/documents/Resultat-PX1.csv")


; lecture fichier
(define liste (read-table (open-input-file fichier-data)))
(close-input-port (open-input-file fichier-data))

;extraction de la colonne de cotation centree, reduite qui servira de reference à la suite
(define colonne2 (centrage-reduction-liste(extract-col col liste)))

; calcul des colonnes pour analyses - elles sont conservées à la longueur iniiale ( avec l'info "indeterminé si necessaire"
;;;;
;pour memoire,
; liste contient la lecture de px1.csv soit les cotations au format ohlc
; colonne contient la colonne et passe au format numerique
; col2 calcule la difference des cours de cloture en (t) et (t+1) INUTILE AVEC LA FONCTION LAG
; MM20 est la moyenne mobile 20 des cours de cloture
; MM50 est la moyenne mobile 50 des cours de cloture
; cloture-lag1 est le cours decalé de + 1 jour ( permettre par difference avec cloture d'obtenir cloture(t-1)-cloture(t) ( hausse ou baisse sur le lag)
; MM20-lag1 MM20 decalée de +1 jour pour permettre d'apprecier l'evolution de MM20 à t
; MM50-lag1 MM20 decalée de +1 jour pour permettre d'apprecier l'evolution de MM50 à t
;;;;
(define MM20(egaliseur-de-liste (length colonne2) (MMx 20 colonne2))); avec respect de la longueur initiale de colonne2
(define MM50 (egaliseur-de-liste (length colonne2) (MMx 50 colonne2))); avec respect de la longueur initiale de colonne2
(define cloture-lag1 (lag -1 colonne2)) ;obtenir la valeur de cloture du lendemain a la date du jour
(define MM20-lag1  (lag 1 MM20))
(define MM50-lag1 (lag 1 MM50))
(define cloture-variation-1 (map -* cloture-lag1 colonne2))

;test des longueurs des listes
;(test-longueur-listes colonne2 cloture-lag1 cloture-variation-1 MM20 MM50 MM20-lag1 MM50-lag1)


;definir la liste des items valides (retenus) pour l'etude (formation et test du reseau)
(define strategie-alpha (validateur colonne2 cloture-lag1 cloture-variation-1 MM20 MM50 MM20-lag1 MM50-lag1))

;extraire des differentes listes les items valides ( au sens de strategie-alpha ) -> pesentation en table des elements
(define liste-valide  (removeAll null (tableur colonne2 MM20 MM50 cloture-variation-1 strategie-alpha)))

;analyse de la table : 1) separation en cluster des données
(define-values (centers clusters) (cluster liste-valide 2))

(printf "centers are about: ~a\n" (map (curry map exact-round) centers))



;definition des bases d'apprentissage et de validation
;;;les hausses
(define base-training-test-h (formateur liste-valide 1))
;;;les hausses retenues pour l'apprentissage
(define base-training-test-h-2 (take(formateur liste-valide 1)  longueur-base-training ))
;;;les autres hausses...pour le test
(define base-test-h (drop base-training-test-h longueur-base-training))
;;;les baisses
(define base-training-test-b(formateur liste-valide 0))
;les baisses retenues pour l'apprentissage
(define base-training-test-b-2 (take base-training-test-b longueur-base-training ))
;;;les autres baisses...pour le test
(define base-test-b (drop base-training-test-b longueur-base-training))

;soit au total l'apprentissage se fait avec :
(define base-training-tot (append base-training-test-h-2 base-training-test-b-2))

; soit au total le test de fait avec :
(define base-test-brute (remove* base-training-tot liste-valide))





"entree en zone neuronale"   ; c'est pour savoir où en est le traitement lors de longues boucles ! )

;creation du reseau ( 3 neurones d'entree, 2 couches masquées de 6 neurones et 1 neurone de sortie (en general, en entree le nbre de colonne d'entree, en sortie le nbre de col de sortie
(define ann (new-ann '(2 8 8 8 8 8 1)))  ;essayer couche intermediaire=2 puissance couche entree

; le reseau fait ses anticipations....mais il n'a pas encore appris
(round-output(run-ann ann '(-0.5278356451551374 -0.7675603963129171 -0.8878725558127648)))   ; hausse realisée
"entrée en zone d'apprentissage"
;apprentissage du reseau avec la base-training;
(do ((i 0 (+ i 1)))
    ((> i 1000) #t)
  
;(train-ann ann '(4322.86 4259.7415 4230.9054) '(1))   pour memoire du programme initial ( sans la boucle map)

  (map(lambda(x)
      (train-ann ann (list (second x)(third x)) (list(* -1 (fifth x))))
        )
    base-training-tot)
  )

" Aprentissage terminé !)"   ; c'est pour savoir où en est le traitement lors de longues boucles ! )
" entrée en zone de validation"
;calculs du reseau apres l'apprentissage,

;(round-output (run-ann ann '(4495.17 4381.096 4317.218))) ;le cac monte ce jour  !     pour memoire du programme initial ( sans la boucle map)

;test des hausses connues
 (define resultat-h(map(lambda(x)
      (run-ann ann (list (second x)(third x)))
                 )
    base-test-h))
;test des baisses connues
 (define resultat-b(map(lambda(x)
      (run-ann ann (list (second x)(third x)))
                 )
    base-test-b))
;test de la base education...toujours interessant de savoir ce que l'on a retenu à la sortie du cours !
(define resultat-f(map(lambda(x)
      (round-output (run-ann ann (list (second x)(third x))))
                 )
    base-training-tot))
"entrée en zone d'analyse des résultats"
; analyse des resultats
;nbre item disponible : (length liste-valide)
;nbre d'item de formation : (length base-training-tot)
; nbre item testés : (length base-test) dont
;                  - nbre de hausses testées : (length base-test-h)  avec pour resultat : resultat-h
;                  - nbre de baisses testées : (length base-test-b)  avec pour resultat : resultat-b

(display " nbre item de formation : ")
(displayln (length base-training-tot))
(display " nbre hausses testées : ") 
(display (length base-test-h))
(display " nbre de hausses detectées : " )   
(displayln (cumul (map (lambda (x) 1) resultat-h)))
(display " nbre baisses testées : ")
(display (length base-test-b))
(display" nbre de baisses detectées : ")
(displayln (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b))))

;si le test est celui des hausses :
;---en ne testant que des hausses avérées je dois trouver des hausses (vp),
;s'il y a un ecart c'est un faux negatif (fn)
;---en ne testant que des baisses averee je dois trouver des baisses (vn),
;s'il y a un ecart c'est une hausse non detectee (fp)
; donc : vp=(cumul (map (lambda (x) 1) resultat-h))
;        fn=(- (length base-test-h) (cumul (map (lambda (x) 1) resultat-h))))
;        vn=(- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b)))
;        fp=(- (length base-test-b) (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b)))))

(display "sensibilite : ")
(displayln (sensibilite (cumul (map (lambda (x) 1) resultat-h)) (- (length base-test-h) (cumul (map (lambda (x) 1) resultat-h)))))


(display "specificite : ")
(displayln (specificite (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b))) (- (length base-test-b) (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b))))))

(display "youden : ")
(displayln (youden
          (sensibilite (cumul (map (lambda (x) 1) resultat-h)) (- (length base-test-h) (cumul (map (lambda (x) 1) resultat-h))))
          (specificite (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b))) (- (length base-test-b) (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b)))))
          ))

; ecriture fichier resultat (fonctionne mais ne gere pas la prexistence du fichier !)
;(define out (open-output-file fichier-sortie))
;(write  (youden
;          (sensibilite (cumul (map (lambda (x) 1) resultat-h)) (- (length base-test-h) (cumul (map (lambda (x) 1) resultat-h))))
;          (specificite (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b))) (- (length base-test-b) (- (length resultat-b) (cumul (map (lambda (x) 1) resultat-b)))))
;          )out)
;(close-output-port out)


;pour info
(displayln "reconstruction base de formation")
 
(map(lambda(x)
      (list (second x)(third x)  (fifth x)))
    base-training-tot)

(displayln resultat-f)


