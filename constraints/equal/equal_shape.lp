#const c = 0.

% :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, Y1 != Y2, S1 != S2, c=0.
% :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, Y1 != Y2, C1 != C2, c=0.

% :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, Y1 != Y2, S1 == S2, c=1.
% :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, Y1 != Y2, C1 == C2, c=1.

negative :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, Y1 != Y2, S1 != S2.

positive :- not negative.
-positive :- negative. 

:- negative, c = 0.
:- positive, c = 1.
