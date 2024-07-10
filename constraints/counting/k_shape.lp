#const c = 1.
#const k = 4.

:- #count{ S : obj(X, Y, S, C)} != k, c = 0.

:- #count{ S : obj(X, Y, S, C)} == k, c = 1.
