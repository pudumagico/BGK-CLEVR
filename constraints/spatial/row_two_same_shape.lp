#const c = 0.

% :- obj(X, Y1, S1, C1), obj(X, Y2, S2, C2), Y1 != Y2, C1 != C2, c=0.

% negative :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, S1 != S2.
% negative :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), X1 != X2, C1 != C2.

negative :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), obj(X3, Y3, S3, C3), X1!=X2, X2!=X3, S1!=S2, S2!=S3.
negative :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), obj(X3, Y3, S3, C3), X1!=X2, X2!=X3, S1==S2, S2==S3.
positive :- obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), obj(X3, Y3, S3, C3), X1!=X2, X2!=X3, S1==S2, S2!=S3.

% :- negative, obj(X1, Y1, S1, C1), obj(X2, Y2, S2, C2), obj(X3, Y3, S3, C3), X1!=X2, S1==S2.

positive :- not negative.
% negative :- not positive.
-positive :- negative. 
% -negative :- positive. 

:- negative, c = 0.
:- positive, c = 1.
