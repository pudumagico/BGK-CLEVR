#const c = 1.

:- obj(X1,Y1,S1,C1), obj(X2,Y2,S2,C2), X1 != X2, Y1 == Y2, C1 != C2, c = 1.
:- #count{ C : obj(X,Y,S,C) } < 2, heigth(Y), c = 0.

