(* ::Package:: *)

ScaledAntiHermitianMatrix[a_]:={{I a,1},{-1,I a}};
JtrigForm[a_,b_,\[Gamma]_] =(MatrixExp[I \[Gamma]/2 KroneckerProduct[ScaledAntiHermitianMatrix[a],ScaledAntiHermitianMatrix[b]]]//TrigToExp)//ExpToTrig//FullSimplify;
J[a_,b_,\[Gamma]_] :=(1/E^(-(1/2) I (1+a) (1+b) \[Gamma]) MatrixExp[I \[Gamma]/2 KroneckerProduct[ScaledAntiHermitianMatrix[a],ScaledAntiHermitianMatrix[b]]]);


MyConj[a_] := ComplexExpand[Conjugate[a]]
MyConj[expr_] := expr/. Complex[a_,b_]-> Complex[a,-b]


play[J_,d1_,d2_]:= ConjugateTranspose[J] . (KroneckerProduct[d1,d2] . (J . {1,0,0,0}))
alicesScore[\[Psi]_]:=((\[Psi]*MyConj[\[Psi]]) . {3,0,5,1});
bobsScore[\[Psi]_]:=((\[Psi]*MyConj[\[Psi]]) . {3,5,0,1});
scores[\[Psi]_]:={alicesScore[\[Psi]],bobsScore[\[Psi]]};
