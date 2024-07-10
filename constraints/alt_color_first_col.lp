% Example 13: The first column must have alternating colors
#const c=1.
:- c=1, obj(0, Y1, _, C1), obj(0, Y2, _, C2), Y1 = Y2 + 1, C1 = C2.

% Example 13 Converse: The first column must not have alternating colors
no_alt_color_first_col :- obj(0, Y1, _, C1), obj(0, Y2, _, C2), Y1 = Y2 + 1, C1 = C2.
:- c=0, not no_alt_color_first_col.