#const c = 1.
#const k = 4.

:- #count{ C : obj(X, Y, S, C) } != k, c = 0.

:- #count{ C : obj(X, Y, S, C) } == k, c = 1.
