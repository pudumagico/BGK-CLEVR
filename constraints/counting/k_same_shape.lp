#const c = 1.
#const k = 4.

:- shape(C), #count{ S : obj(X, Y, S, C)} != k, c = 0.

:- shape(C), #count{ S : obj(X, Y, S, C)} == k, c = 1.
