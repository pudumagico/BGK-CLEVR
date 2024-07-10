#const c = 1.
#const k = 4.

:- shape(S), #count{ C : obj(X, Y, S, C) } != k, c = 0.

:- shape(S), #count{ C : obj(X, Y, S, C) } == k, c = 1.
