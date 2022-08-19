#const c = 1.

:- obj(X1,Y1,S1,C1), obj(X2,Y2,S2,C2), X1 != X2, Y1 == Y2, S1 != S2, c = 1.
:- #count{ X,S : width(X), obj(X,Y,S,C) } < 1, c = 0.
