(* ::Package:: *)

(*Main definition for a list with one or more elements*)
SymbolicMax[exprList_List]:=Module[{n=Length[exprList],conditionsPairs},
conditionsPairs=Table[
{exprList[[i]],And@@Table[exprList[[i]]>=exprList[[j]],
{j,Complement[Range[n],{i}]}]}
,{i,n}];

conditionsPairs];

(*Main definition for a list with one or more elements*)
SymbolicArgMax[exprList_List]:=Module[{n=Length[exprList],conditionsPairs},
conditionsPairs=Table[
{i,And@@Table[exprList[[i]]>=exprList[[j]],
{j,Complement[Range[n],{i}]}]}
,{i,n}];

conditionsPairs];


DiscreteNashEquilibria[bscores_, ascores_,conditions_] := Module[{
  nn = Length[ascores],
  p1Best, p2Best, equilibria,
  progress = 0 (* Initialize a progress counter *)
  },

 (* p1Best[[i]] gives Player 2's best response to Player 1 playing i *)
 p1Best = SymbolicArgMax /@ ascores;
 (* p2Best[[j]] gives Player 1's best response to Player 2 playing j *)
 p2Best = SymbolicArgMax /@ Transpose[bscores];

 (* Monitor the Table computation *)
 equilibria = Monitor[
   Flatten[ParallelTable[
     Module[{cond1, cond2, combinedCond},
      cond1 = FirstCase[p1Best[[i]], {j, c_} :> c, False];
      If[cond1 === False, Nothing,
       cond2 = FirstCase[p2Best[[j]], {i, c_} :> c, False];
       If[cond2 === False, Nothing,
        combinedCond = cond1 && cond2;
        combinedCond = TimeConstrained[Reduce[conditions&&combinedCond , {a, \[Gamma]}, Reals],30,combinedCond];
        If[combinedCond === False, Nothing,
         {{i, j}, {ascores[[i, j]], bscores[[i, j]]}, combinedCond}
         ]
        ]
       ]
      ],
     {i, 1, nn}, {j, 1, nn}], 1],
   ];

 Return[equilibria];
 ]
