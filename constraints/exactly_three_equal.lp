% Constraint: Exactly three objects must be equal in shape and color
three_equal(S, C) :- obj(X1, Y1, S, C), obj(X2, Y2, S, C), obj(X3, Y3, S, C), (X1, Y1) != (X2, Y2), (X1, Y1) != (X3, Y3), (X2, Y2) != (X3, Y3).
more_than_three_equal(S, C) :- obj(X1, Y1, S, C), obj(X2, Y2, S, C), obj(X3, Y3, S, C), obj(X4, Y4, S, C), (X1, Y1) != (X2, Y2), (X1, Y1) != (X3, Y3), (X1, Y1) != (X4, Y4), (X2, Y2) != (X3, Y3), (X2, Y2) != (X4, Y4), (X3, Y3) != (X4, Y4).

% Exactly two objects must be equal in shape and color
:- c=1, not three_equal(_, _).
:- c=1, more_than_three_equal(S, C).

% Constraint: There must not be exactly two objects that are the same
:- c=0, three_equal(_, _), not more_than_three_equal(_, _).


