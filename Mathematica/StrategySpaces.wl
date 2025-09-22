(* ::Package:: *)

(* ::Subsection:: *)
(*Strategy Spaces*)


(* ::Subsubsection:: *)
(*su2*)


(* ::Text:: *)
(*An arbitrary element of su2, where \[Theta] and \[Phi] define a unit vector, and \[Zeta] a rotation around that unit vector.*)


su2[\[Theta]_,\[Phi]_,\[Zeta]_]:={{Cos[\[Zeta]/2]-I*Cos[\[Theta]]*Sin[\[Zeta]/2],Sin[\[Zeta]/2]*Sin[\[Theta]]*((-I)*Cos[\[Phi]]-Sin[\[Phi]])},{Sin[\[Zeta]/2]*Sin[\[Theta]]*((-I)*Cos[\[Phi]]+Sin[\[Phi]]),Cos[\[Zeta]/2]+I*Cos[\[Theta]]*Sin[\[Zeta]/2]}}
su2[\[Phi]p_,\[Phi]m_,\[Theta]_]:= RotationMatrix[\[Theta]]*{{Exp[\[Phi]p I],Exp[\[Phi]m I]},{Exp[- \[Phi]m I],Exp[-\[Phi]p I]}}


(* ::Subsubsection::Closed:: *)
(*cyclic*)


cyclic[k_,n_]:={{Cos[2\[Pi] k/n],-Sin[2\[Pi] k/n]},{Sin[2\[Pi] k/n],Cos[2\[Pi] k/n]}};
cyclicGroup[n_] /;n>2:=Table[cyclic[k,n],{k,0,n}]


(* ::Subsubsection:: *)
(*binary Dihedral*)


(*---single element of the binary dihedral group Dic_n---*)
binaryDihedral[k_,x_,n_]:=Module[{A,B},(*A=rotation by \[Pi]/n about z:order 2n,with A^n=-I*)
A= {{Exp[I*\[Pi]*k/n],0},{0,Exp[-I*\[Pi]*k/n]}};
(*B=lift of the 180\[Degree] flip about x:B^2=-I=A^n*)B={{0,1},{-1,0}};
A . MatrixPower[B,x]//FullSimplify];

(*---assemble all 4n elements and drop duplicates---*)
binaryDihedralGroup[n_]:=Module[{elems},elems=Flatten[Table[binaryDihedral[k,x,n],{k,0,2 n-1},(*A^0 \[Ellipsis] A^(2n\[Minus]1)*){x,0,1}         (*identity or one B-flip*)],1];
DeleteDuplicates[elems]];


(* ::Subsubsection:: *)
(*Pin*)


(*---single element of the binary dihedral group Dic_n---*)
pin[\[Theta]_,x_]:=Module[{A,B},(*A=rotation by \[Pi]/n about z:order 2n,with A^n=-I*)
A={{Exp[I \[Theta]],0},{0,Exp[-I \[Theta]]}};
(*B=lift of the 180\[Degree] flip about x:B^2=-I=A^n*)B={{0,1},{-1,0}};
A . MatrixPower[B,x]//FullSimplify];


(* ::Subsubsection:: *)
(*binary tetrahedral (incorrect becouse you need srs)*)


binaryTetrahedral[i_,j_]:=Module[{\[Omega]1=Exp[2 Pi I/6],\[Omega]2=Exp[Pi I/2],A,B},
A={{\[Omega]1,0},{0,Conjugate[\[Omega]1]}};
B=(1/Sqrt[2]) {{\[Omega]2,\[Omega]2},{\[Omega]2,-\[Omega]2}};
MatrixPower[A,i] . MatrixPower[B,j]]

(*Function to generate the full group*)binaryTetrahedralGroup=Module[{elements},elements=Flatten[Table[binaryTetrahedral[i,j],{i,0,5},(*A has order 6*){j,0,3}   (*B has order 4*)],1];
DeleteDuplicates[elements]];


(* ::Subsubsection::Closed:: *)
(*binary octahedral (incorrect becouse you need srs)*)


(*single binary-octahedral element from exponents i,j*)binaryOctahedral[i_,j_]:=Module[{\[Omega]8=Exp[I*\[Pi]/4],(*e^{i\[Pi]/4}*)r,s},r={{\[Omega]8,0},{0,Conjugate[\[Omega]8]}};(*order 8*)s=1/2 {{1+I,1+I},{I-1,1-I}};(*order 6*)MatrixPower[r,i] . MatrixPower[s,j]];

(*the full group of 48 elements*)
binaryOctahedralGroup=Module[{elems},elems=Flatten[Table[binaryOctahedral[i,j],{i,0,7},(*r^8=I*){j,0,5}     (*s^6=I*)],1];
DeleteDuplicates[elems]];

