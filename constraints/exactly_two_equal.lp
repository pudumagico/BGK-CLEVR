% Constraint: Exactly two objects must be equal in shape and color
#const c=1.
two_equal(S, C) :- obj(X1, Y1, S, C), obj(X2, Y2, S, C), (X1, Y1) != (X2, Y2).
more_than_two_equal(S, C) :- obj(X1, Y1, S, C), obj(X2, Y2, S, C), obj(X3, Y3, S, C), (X1, Y1) != (X2, Y2), (X1, Y1) != (X3, Y3), (X2, Y2) != (X3, Y3).

% Exactly two objects must be equal in shape and color
:- c=1, not two_equal(_, _).
:- c=1, more_than_two_equal(S, C).

% Constraint: There must not be exactly two objects that are the same
:- c=0, two_equal(_, _), not more_than_two_equal(_, _).

