from cmath import inf
from telnetlib import GA
import numpy as np
import random
import math
from operator import concat
import matplotlib.pyplot as plt
from itertools import combinations
import pulp
import pandas as pd
#Declaration des variables globales:
N=6
INF=100000000000
timestep=1
timeLimit=100
roundDigit=10
randomIntChooser = random.SystemRandom()
M_big=INF
epsilon=2

#Declaration des variables uniques aux problèmes:


#m=[120,150,90,120,90,30,60,60,45,130,120,90,30,0]
#m=[360,60,30,120,180,30,30,60,60,300,60,60,120,0,0]
#m=[0,56,56,143,127,137,148,112,86,144,72,79,128,0]
#M=[INF,68,86,146,140,184,193,133,129,164,78,110,140]
#M=[INF,200,120,180,125,40,120,120,75,INF,INF,120,60,INF]
#M=[INF,120,90,240,240,90,90,120,120,420,120,120,180,INF]
#r=[31,22,22,22,25,23,22,22,22,47,27,22,30,0]
#r=[23,29,19,24,19,19,24,24,20,24,24,19,30,23]
'''d=[[ 0, 11, 14,16,14,19,22,24,26,29,6,8,10,0],
   [11,  0, 2,5,2,8,10,13,15,17,10,3,10,11],
   [14,  2, 0,2,0,5,8,10,13,15,12,6,3,14],
   [16,  5, 2,0,2,3,5,8,10,13,15,8,6,16],
   [14,  2, 0,2,0,5,8,10,13,15,12,6,3,16],
   [19,  8, 5,3,5,0,3,5,7,10,18,11,9,19],
   [22, 10, 8,5,8,3,0,2,5,7,20,14,11,22],
   [24, 13, 10,8,10,5,2,0,2,5,23,16,14,24],
   [26, 15, 13,10,13,7,5,2,0,2,25,19,16,26],
   [29, 17, 15,13,15,10,7,5,2,0,27,21,19,29],
   [ 6, 10, 12,15,12,18,20,23,25,27,0,7,9,6],
   [ 8,  3, 6,8,6,11,14,16,19,21,7,0,2,8],    
   [10, 1, 3,6,3,9,11,14,16,19,9,2,0,10],
   [0,11,14,16,14,19,22,24,26,29,6,8,10,0]]'''
'''
d=[[0,13,12,11,10,9,9,8,8,6,6,5,5,0],
   [13,0,4,5,6,7,7,8,9,10,10,11,12,13],
   [12,4,0,4,5,6,6,7,8,9,9,10,11,12],
   [11,5,4,0,4,5,6,6,7,8,9,10,10,11],
   [10,6,5,4,0,4,5,6,6,7,8,9,9,10],
   [9,7,6,5,4,0,4,5,5,6,7,8,8,9],
   [9,7,6,6,5,4,0,4,4,5,6,7,8,9],
   [8,8,7,6,6,5,4,0,4,4,5,6,7,8],
   [8,9,8,7,6,5,4,4,0,5,5,6,7,8],
   [6,10,9,8,7,6,5,4,5,0,4,5,5,6],
   [6,10,9,9,8,7,6,5,5,4,0,4,5,6],
   [5,11,10,10,9,8,7,6,6,5,4,0,4,5],
   [5,12,11,10,9,8,8,7,7,5,5,4,0,5],
   [0,13,12,11,10,9,9,8,8,6,6,5,5,0]]   '''

#Variables du problème conçu:

m=[10,50,30,10,10,50,0]
M=[INF,100,120,50,100,100,100]
sigma=[1,2,3,1,1,2,2,1,1,1,1,2,1]
d=[]
for i in range(N):
    d.append([])
    for j in range(N):
        d[i].append(0)
for i in range(N):
    for j in range(N):
        if j>i:
            d[i][j]=sum(sigma[k] for k in range(i,j))
        elif i>j:
            d[i][j]=sum(sigma[k] for k in range(j,i))
        else:
            d[i][j]=0
for i in range(N):
    for j in range(N):
        if j==N-1:
            print(d[i][j],"\\", "\n")
        if j==0:
            print(i, "&",d[i][j],end=" ")
        else:
            print(d[i][j], " &",end=" ")

r=[21,22,23,21,21,22,22,21,21,21,21,22,21]

#Definition de la classe robot:
class Robot:
    def __init__(self, index, moveSequenceList, startTimeSequenceList, currPos, nextPos):
        self.index=index
        self.currPos=currPos
        self.nextPos=nextPos
        self.moveSequenceList=moveSequenceList
        self.startTimeSequenceList=startTimeSequenceList
        self.inMoveVide=False 
        self.inMoveCharge=True
        self.currIndex=1
        self.fullMoveSequenceList=[]
    def toString(self):
        return "Robot indexed: ",self.index, "at position: " ,self.currPos ,"going to: " ,self.nextPos,"moveInVide", self.inMoveVide, "moveInCharge", self.inMoveCharge, "timeStartList:", self.startTimeSequenceList, "movementList:", self.moveSequenceList


def fromSeqCreateSimulation(Solution):
    robotList=[]
    for robotIterator in range(1,Solution[1]+1):
        robotList.append(Robot(robotIterator,Solution[0][robotIterator-1],Solution[robotIterator+2],Solution[0][robotIterator-1][0],Solution[0][robotIterator-1][1]))
        print(robotList[robotIterator-1].toString())
    return robotList

def fromVSeqCreateSimulation(Solution):
    robotList=[]
    for robotIterator in range(1,Solution[1]+1):
        L=[]
        for j in range(len(Solution[0][robotIterator-1])-1):
                if j%2==1:
                    L.append([Solution[0][robotIterator-1][j],Solution[0][robotIterator-1][j+1]])
        indexes=[i[0] for i in sorted(enumerate(Solution[robotIterator+2]), key=lambda x:x[1])]
        L=[L[i] for i in indexes]
        Solution[robotIterator+2].sort()
        robotList.append(Robot(robotIterator,L,Solution[robotIterator+2],L[0][0]-1,L[0][0]))
        print(robotList[robotIterator-1].toString())
    return robotList


#Fonction MNS:
def MNS(N):
    return math.ceil((N-3)/2)

#L'insertion des séparateurs dans une solution:
def separatorInsertion(N, O):
    NS=randomIntChooser.randint(1,MNS(N))
    for separatorIterator in range(1,NS+1):
        O.insert(randomIntChooser.randint(0,len(O)),0)
    return NS

#Géneration d'une solution à partir des nombres de cuves N:
def solutionGeneration(N):
    NC=randomIntChooser.randint(6,N)
    #print("NC=",NC)
    O=[]
    for numberOfTanksIterator in range(1,NC+1):
        randomTankNumber=randomIntChooser.randint(1,N)
        while randomTankNumber in O:
            randomTankNumber = randomIntChooser.randint(1,N)
        O.append(randomTankNumber)
    NS=separatorInsertion(N,O)
    #print("O=",O)
    return [O,NC,NS]

#Réparation des insertions des séparateurs
def solutionSeparatorRepair(O,NC,NS):
    P=[]
    toRemove=[]
    O_prime=[]
    k=0
    for i in range(len(O)):
        if O[i]==0:
            P.append(i)
        else:
            O_prime.append(O[i])
    #print("P0=",P)
    for i in range(0,NS):
        P[i]=P[i]-1-(i-1)
        O.pop(P[i])
    #print("P1=",P)
    for i in range(len(P)-1):
        if P[i]==P[i+1] or abs(P[i]-P[i+1])<2:
            toRemove.append(P[i+1])
    for i in range(len(P)):
        if (P[i]==0) or (P[i]==1) or (P[i]==NC) or (P[i]==NC-1):
            toRemove.append(P[i])
    #print("toRemove= ", toRemove)
    for i in range(len(toRemove)):
        if toRemove[i] in P:
            P.remove(toRemove[i])
    for i in range(len(P)):
        O.insert(P[i]+k,0)
        k=+1
    NS=len(P)
    NC=len(O_prime)
    #print("P=",P)
    return [NS,NC,P,O_prime,O]

#Décodage d'une solution (Algo figurant dans la thèse):
def solutionDecoding(NS, NC, P, O_prime):
    h=1
    Seq=[]
    Sh=[]
    z=[0]
    Sh.append(1)
    Sh.append(2)
    k=1
    d=0
    LMO=list(range(1,N+1))
    LMO=list(range(2,N+1))
    #print(LMO)
    while(d<N):
        #print("///////////////////////////")
        #print("Sh=",Sh)
        #print("d=", d)
        if Sh[k] in O_prime:
            POS=O_prime.index(Sh[k])+1
        else:
            POS=-1
        #print("POS=",POS)
        if POS==-1:
            Sh.append(Sh[k])
        else:
            if POS in P:
                if POS==P[0]:
                    Sh.append(O_prime[0])
                    
                else:
                    s=P.index(POS)
                    Sh.append(O_prime[P[s-1]])
            else:
                if POS==NC:
                    if NS>0:
                        Sh.append(O_prime[P[NS-1]])
                    else:
                        Sh.append(O_prime[0])
                else:
                    #print("OO POS==", POS, O_prime)
                    Sh.append(O_prime[POS])
        if Sh[k+1] in LMO:
            LMO.remove(Sh[k+1]) 
        d=d+1
        #print("("+str(Sh[k+1])+","+str(Sh[0])+")")
        if Sh[k+1]==Sh[0]:
            Seq.append(Sh)
            z.append(math.floor(len(Sh)/2))
            #print("concat done")
            if d <= N:
                Sh=[]
                if LMO:
                    h=h+1
                    Sh.append(min(LMO))
                    Sh.append(Sh[0]+1)
                    k=1
        else:
            if Sh[k+1]==N:
                Sh.append(1)
            else:
                Sh.append(Sh[k+1]+1)
            k=k+2
        #print("Seq=",Seq)
        
    H=len(Seq)
    return [Seq, H,z]

#Evaluation d'une séquence en utilisant un programme linéare:
def evaluation(Seq,H,z):
    file= open("C:\\Users\\RafikMedici\\Desktop\\empty.txt", 'w')
    #get sequence
    #print("Seq,H,z=",Seq,H,z)
    if Seq==[]:
        return 10000
    L=[]
    L_r=[]
    L_rr=[]
    for i in range(len(Seq)):
        L_r.append([])
        L_rr.append([])
        for j in range(len(Seq[i])-1):
            if j%2==1:
                L.append((Seq[i][j],Seq[i][j+1]))
                L_r[i].append((Seq[i][j],Seq[i][j+1]))
                L_rr[i].append(Seq[i][j])
    #print("L=",L)
    # Create a new optimization problem
    problem = pulp.LpProblem("Minimize_T", pulp.LpMinimize)

    # Define decision variables
    T = pulp.LpVariable("T", lowBound=0)
    t = [pulp.LpVariable(f"t{i}",lowBound=0) for i in range(0, N)]
    b = [pulp.LpVariable(f"b{i}",lowBound=0,upBound=1, cat=pulp.LpBinary) for i in range(0, N)]
    a = [[pulp.LpVariable(f"a{h},{u}",lowBound=0,upBound=1, cat=pulp.LpBinary) for u in range(1, z[h]+1)] for h in range(1, H+1)]

    # Add objective function
    problem += T #(2.1)

    # Add constraints
    for i in range(0, N):
        problem += t[i] <= T #(2.2)
        problem += t[i]>=0
        ourstr="t("+str(i+1)+")<=T\n"
        file.writelines(ourstr)
    problem += t[1] == r[0]  #(2.3)
    #for i in range(1,len(Seq)):
        #problem += t[Seq[i][0]] == r[Seq[i][0]]
    ourstr="t(2)==r(1)=="+str(r[0])+"\n"
    #print(ourstr)
    file.write(ourstr)
    problem += t[0] + m[N] <= T #(2.4)
    ourstr="t(1)+m("+str(N+1)+")<=T"+", m(14)="+str(m[N])+"\n"
    file.write(ourstr)

    for i in range(0, N):
        problem += m[i] <= T #(2.5)
        ourstr="m("+str(i+1)+")<=T"+ ", m("+str(i+1)+")="+str(m[i])+"\n"
        file.write(ourstr)
    for i in range(0, N-1):
        problem +=  t[i] - (t[i+1] - r[i]) <= b[i] * M_big #(2.6)
        problem +=  t[i] - (t[i+1] - r[i])>=(b[i] - 1) * M_big
        ourstr="(b("+str(i+1)+")- 1) * M_big <= t("+str(i+1)+")  - (t("+str(i+2)+")  - r("+str(i+1)+")) <= b("+str(i+1)+")  * M_big"+"\n"
        file.write(ourstr)
    #print("r[N-1]=", r[N-1])
    problem += (b[N-1] - 1) * M_big<= t[N-1] - (t[0] - r[N-1]) <= b[N-1] * M_big #(2.7)
    problem += t[N-1] - (t[0] - r[N-1]) >=(b[N-1] - 1) * M_big
    ourstr="(b("+str(N)+") - 1) * M_big<= t("+str(N)+") - (t("+str(1)+") - r("+str(N)+")) <= b("+str(N)+") * M_big"+"\n"
    file.write(ourstr)
    
    for i in range(0, N-1):
        problem += m[i] - b[i] * M_big <= t[i+1] - r[i] - t[i] <=M[i] + b[i] * M_big #(2.8)
        problem += t[i+1] - r[i] - t[i]>=m[i] - b[i] * M_big 
        ourstr="m("+str(i+1)+") - b("+str(i+1)+") * M_big <= t("+str(i+2)+") - r("+str(i+1)+") - t("+str(i+1)+") <=M("+str(i+1)+") + b("+str(i+1)+") * M_big"+"\n"
        file.write(ourstr)
    for i in range(0,N-1):
        problem += m[i] + (b[i]-1) * M_big <= T+t[i+1] - r[i] - t[i] <= M[i] + (1-b[i]) * M_big #(2.9)
        problem += T+t[i+1] - r[i] - t[i]>=m[i] + (b[i]-1) * M_big
        ourstr="m("+str(i+1)+") + (b("+str(i+1)+")-1) * M_big <= T+t("+str(i+2)+") - r("+str(i+1)+")- t("+str(i+1)+") <= M("+str(i+1)+") + (1-b("+str(i+1)+")) * M_big"+"\n"
        file.write(ourstr)
    problem += t[0] - r[N-1] - t[N-1] <= M[N-1] + b[N-1]* M_big #(2.10)
    problem +=  t[0] - r[N-1] - t[N-1]>=m[N-1] - b[N-1] * M_big 
    ourstr="m("+str(N)+") - b("+str(N)+") * M_big <= t("+str(1)+") - r("+str(N)+") - t("+str(N)+") <= M("+str(N)+") + b("+str(N)+")* M_big"+"\n"
    file.write(ourstr)
    problem +=  T+t[0] - r[N-1] - t[N-1] <= M[N-1] + (1-b[N-1])* M_big #(2.11
    problem +=  T+t[0] - r[N-1] - t[N-1]>=m[N-1] + (b[N-1]-1) * M_big 
    ourstr="m("+str(N)+")  + (b("+str(N)+")-1) * M_big <= T+t("+str(1)+") - r("+str(N)+") - t("+str(N)+")<= M("+str(N)+") + (1-b("+str(N)+"))* M_big"+"\n"
    file.write(ourstr)
    for h in range(1,H+1):
        problem+=sum(a[h-1][u-1] for u in range(1, z[h]+1))==1 #(2.12)
    
    for u in range(1, z[1]):
        problem += a[0][u-1] == 0
    problem += a[0][z[1]-1] == 1 #(2.13)
    ourstr="a(1)"+"("+str(z[1])+")==1"+"\n"
    file.write(ourstr)
    
    
    for h in range(1, H+1):
        for u in range(1, z[h]+1):
            j=L_r[h-1][u-1][0]
            i=L_r[h-1][u-1][1]
            print("(j,i)=(",j,",",i,")")
            if (not i==N):
                    problem += t[j-1] + d[j-1][i-1] <= t[i] - r[i-1] + a[h-1][u-1] * M_big #(2.14)
                    ourstr="t("+str(j+1)+") + d("+str(j+1)+")("+str(i+1)+") <= t("+str(i+2)+") - r("+str(i+1)+") + a("+str(h)+")("+str(u)+") * M_big"+"\n"
                    file.write(ourstr)
                    problem += t[j-1] + d[j-1][i-1] <= T + t[i] - r[i-1] + (1 - a[h-1][u-1]) * M_big #(2.15)
                    ourstr="t("+str(j+1)+") + d("+str(j+1)+")("+str(i+1)+") <= T +  t("+str(i+2)+") - r("+str(i+1)+") + (1 - a("+str(h)+")("+str(u)+")) * M_big"+"\n"
                    file.write(ourstr)
            elif i==N:
                    problem += t[j-1] + d[j-1][N-1] <= t[0] - r[N-1] + a[h-1][u-1] * M_big #(2.14)
                    ourstr="t("+str(j+1)+") + d("+str(j+1)+")("+str(N)+") <= t("+str(1)+") - r("+str(N)+") + a("+str(h)+")("+str(u)+") * M_big"+"\n"
                    file.write(ourstr)
                    problem += t[j-1] + d[j-1][N-1] <= T + t[0] - r[N-1] + (1 - a[h-1][u-1]) * M_big #(2.15)
                    ourstr="t("+str(j+1)+") + d("+str(j+1)+")("+str(N)+") <= T +  t("+str(1)+")  - r("+str(N)+") + (1 - a("+str(h)+")("+str(u)+")) * M_big"+"\n"
                    file.write(ourstr)


    # Solve the problem

    problem.solve()
    # Print the results
    #print(problem)
    print("Status:", pulp.LpStatus[problem.status])
    print("Objective:", pulp.value(problem.objective))
    T_r=[]
    for v in problem.variables():
        print(v.name, "=", v.varValue)
    #for (i,j) in L:
        #print("t(",i,")=", t[i-1])
    for i in range(len(L_rr)):
        T_r.append([])
        for j in range(len(L_r[i])):
            T_r[i].append(int(t[L_rr[i][j]-1].varValue))
    #print("Sq=", Seq)
    #print("Lrr=", L_rr)
    #print("T_r=", T_r)
    if (pulp.LpStatus[problem.status]=="Infeasible"):
        return [[150000]]
    else:
        #for timeList in T_r:
            #timeList.sort()
        return [T_r,T.varValue]

#Fonctions utilisées pour le traçage des diagramme GANTT des solutions:

#Obtention des a,b des droites construites par les deux points de départ et arrivée:
def getAandB(x,k,X,Y):
    i=0
    for j in range(len(X[k])):
        if x<=X[k][j] and j<len(X[k])+1:
            i=j-1
            break
    x1=X[k][i]
    x2=X[k][i+1]
    y1=Y[k][i]
    y2=Y[k][i+1]
    a=(y1-y2)/(x1-x2)
    b=(y1*x2-y2*x1)/(x2-x1)
    #print("x=",x,",x1=",x1,",x2=",x2,",y1=",y1,",y2=",y2,",a=",a,",b=",b)
    return a, b
#Fonction ax+b:
def ourFunction(x,k,X,Y):
    a,b=getAandB(x,k,X,Y)
    return a*x+b

#La fonction objective à minimiser dans l'algo génetique:
def objective(repairedSol):
    NS=0
    P=[]
    i=0
    while i< len(repairedSol):
        if repairedSol[i]==0:
            NS+=1
            P.append(i)
            repairedSol.remove(repairedSol[i])
        else :
            i+=1

    NC=len(repairedSol)
    decodedSol=solutionDecoding(NS,NC,P,repairedSol)
    return (1/evaluation(decodedSol[0], decodedSol[1], decodedSol[2]))



#Ajout des temps de début des mouvements après évaluation:
def addStartTimeListToDecodedSolution(decodedSolution):
    OTK=evaluation(decodedSolution[0], decodedSolution[1], decodedSolution[2])[0]
    feasible=False
    if not (OTK[0]==150000):
        feasible=True
        #print("decodedSolution[1]=", decodedSolution[1])
        #print("OTK=", OTK)
        for j in range(decodedSolution[1]):
                decodedSolution.append(OTK[j])
    return [feasible, decodedSolution]

#Obtention des vecteurs X,Y à partir d'une séquence:
def getXandYfromSequence(robotList):
    Y=[]
    time_x=[]
    for robot in robotList:
        if robotList.index(robot)==0:
            Y.append([1])
        else:
            Y.append([robot.moveSequenceList[0][0]])
        time_x.append([0])
        mv=0
        for move in robot.moveSequenceList:
            mv+=1
            if (move[0]==1):
                Y[robotList.index(robot)].append(N+1)
            else:
                Y[robotList.index(robot)].append(move[0])
                
            if robotList.index(robot)!=0 and time_x[robotList.index(robot)][len(time_x[robotList.index(robot)])-1]==0:
                time_x[robotList.index(robot)].append(0)
            else:
                time_x[robotList.index(robot)].append(r[move[0]-2]+time_x[robotList.index(robot)][len(time_x[robotList.index(robot)])-1])
                
            if (move[0]==1):
                Y[robotList.index(robot)].append(N+1)
            else:
                Y[robotList.index(robot)].append(move[0])
                
            time_x[robotList.index(robot)].append(robot.startTimeSequenceList[robot.moveSequenceList.index(move)])
            
            
            if (robot.moveSequenceList.index(move)==len(robot.moveSequenceList)-1 and move[1]==1 ):
                Y[robotList.index(robot)].append(1)
            else:
                Y[robotList.index(robot)].append(move[1])
            if (not True):
            #if (move[1]==move[0]) and robot.moveSequenceList.index(move)!=len(robot.moveSequenceList)-1:
                time_x[robotList.index(robot)].append(robot.startTimeSequenceList[robot.moveSequenceList.index(move)+1]-r[move[1]-1])
            else:
                time_x[robotList.index(robot)].append(robot.startTimeSequenceList[robot.moveSequenceList.index(move)]+d[move[0]-1][move[1]-1])
    
    for robot in robotList:
        if robot.moveSequenceList[len(robot.moveSequenceList)-1][1]!=1: 
            Y[robotList.index(robot)].append(robot.moveSequenceList[len(robot.moveSequenceList)-1][1]+1)
            time_x[robotList.index(robot)].append(max(time_x[robotList.index(robot)])+r[robot.moveSequenceList[len(robot.moveSequenceList)-1][1]-1])
        '''print("Y=", Y[robotList.index(robot)])
        print("len(Y)=", len(Y))
        print("lethis:", robotList.index(robot), "is le :", Y[robotList.index(robot)][len(Y[robotList.index(robot)])-1])'''
        Y[robotList.index(robot)].append(Y[robotList.index(robot)][len(Y[robotList.index(robot)])-1])
        '''print("Y=", Y[robotList.index(robot)])
        print("lethis:", robotList.index(robot), "is le :", Y[robotList.index(robot)][len(Y[robotList.index(robot)])-1])'''
        if len(robotList)>1:
            time_x[robotList.index(robot)].append(max(max(time_x[0]), max(time_x[1])))
        else:
            time_x[robotList.index(robot)].append(max(time_x[0]))
    '''df = pd.DataFrame({'col': time_x})
    df.drop_duplicates(inplace=True)
    time_x= df['col'].tolist()
    df = pd.DataFrame({'col': Y})
    df.drop_duplicates(inplace=True)
    Y= df['col'].tolist()'''
    print("time_x=", time_x)
    print("Y=", Y)
    return time_x,Y

#Traçage du GANTT:
def graph(X,Y):
    colorList=['r', 'b', 'g']
    
    for k in range(len(X)):
        colorIdx=random.randint(0,2)
        K=[k]*len(X[k])
        X_=[X]*len(X[k])
        Y_=[Y]*len(X[k])
        x=np.array(X[k]) 
        y=np.array(list(map(ourFunction,x,K,X_,Y_)))
        for i in range(len(X[k])-2):
            #print("y(i)=",y[i],"y(i+1)=",y[i+1])
            if -0.1<y[i]-y[i+1]+1<0.1 and -0.1<y[i+1]-y[i+2]<0.1 :
                plt.plot(x[i:i+2],y[i:i+2],ls = '-',marker='o', color=colorList[colorIdx])
                #print("idx=", i)
            else:
                plt.plot(x[i:i+2],y[i:i+2],ls = ':', marker='o',color=colorList[colorIdx])
                #print("idx2=", i)
        plt.plot(x[len(x)-2:],y[len(y)-2:],ls = ':', marker='o',color=colorList[colorIdx])
    plt.grid()
    #plt.savefig("C:\\Users\\RafikMedici\\Desktop\\ourExamples\\example"+str(n)+".png")
    plt.show()

#Obtention des temps de trempe pour assurer le non-viol des contraintes temporelles du problème:
def getTrempeTime(robotList, ET):
    trempeTime=[0]*13
    for robot in robotList:
        for move in robot.moveSequenceList:
            i=move[0]
            if(i!=1):
                k=0
                l=0
                robotidx1=-1
                robotidx2=-1
                for robot_ in robotList:
                    for move in robot_.moveSequenceList:
                        #print("move=", move)
                        if move[1]==i-1:
                            k=move[0]
                            robotidx1=robotList.index(robot_)
                            #print("movefound=,[",k,",",i-1,"]")
                        if move[1]==i:
                            l=move[0]
                            robotidx2=robotList.index(robot_)
                            #print("movefound=,[",l,",",i,"]")
                if  (robotList[robotidx1].moveSequenceList.index([k,i-1])==len(robotList[robotidx1].moveSequenceList)-1):
                    Tk=0
                    Tl=ET[0][robotidx2][robotList[robotidx2].moveSequenceList.index([l,i])]
                    TrempeI=Tl+d[l-1][i-1]-Tk-r[i-2]
                    
                else:
                    Tk=ET[0][robotidx1][robotList[robotidx1].moveSequenceList.index([k,i-1])]
                    Tl=ET[0][robotidx2][robotList[robotidx2].moveSequenceList.index([l,i])]
                    TrempeI=Tl+d[l-1][i-1]-Tk-d[k-1][i-2]-r[i-2]
                if TrempeI>=0:
                    trempeTime[i-1]=TrempeI
                else:
                    trempeTime[i-1]=TrempeI+ET[1]
            else:
                print("uwu")
    return trempeTime

#Obtention des séquences à partir de la liste des object "robot"!
def getSeqHZfromY(robotList):
    Seq=[]
    H=len(robotList)
    z=[0]
    for robot in robotList:
        z.append(len(robot.moveSequenceList))
    for robot in robotList:
        Seq.append([])
        Seq[robotList.index(robot)].append(robot.moveSequenceList[0][0]-1)
        for move in robot.moveSequenceList:
            Seq[robotList.index(robot)].append(move[0])
            Seq[robotList.index(robot)].append(move[1])
    return [Seq,H,z]


#Generation d'une population:
def getPopulation():
    population=[]
    while  (len(population)<20):
        sol=solutionGeneration(N)
        repairedSol=solutionSeparatorRepair(sol[0],sol[1],sol[2])    
        decodedSol=solutionDecoding(repairedSol[0],repairedSol[1],repairedSol[2],repairedSol[3])
        if (decodedSol[1]==2 and decodedSol[2][1]==3):
            OT=evaluation(decodedSol[0],decodedSol[1],decodedSol[2])
            if not (OT[0][0]==150000):
            #T.append(OT[1])
                for j in range(decodedSol[1]):
                    decodedSol.append(OT[0][j])
            #Tr=getTrempeTime(robotList, OT)
            #print("TrempeTime= ",Tr)
            #for i in range(len(Tr)):
            #print(m[i],"<",Tr[i] ,"<",M[i])
                print("decodedSol=", decodedSol)
                newSol=repairPopulation(decodedSol)
                OT=evaluation(newSol[0], newSol[1], newSol[2])
                if newSol!=decodedSol:
                    decodedSol.pop()
                    decodedSol.pop()
                    population.append([decodedSol,newSol])
                
    return population




#Generation d'une population sans réparation:
def getPopulationNonRepare():
    population=[]
    while  (len(population)<20):
        sol=solutionGeneration(N)
        repairedSol=solutionSeparatorRepair(sol[0],sol[1],sol[2])    
        decodedSol=solutionDecoding(repairedSol[0],repairedSol[1],repairedSol[2],repairedSol[3])
        if (decodedSol[1]==2 and decodedSol[2][1]==3):
            OT=evaluation(decodedSol[0],decodedSol[1],decodedSol[2])
            if not (OT[0][0]==150000):
            #T.append(OT[1])
                for j in range(decodedSol[1]):
                    decodedSol.append(OT[0][j])
                    population.append(repairedSol[4])
    return population 



#Algo génetique:       
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = random.randint(0,len(pop)-1)
    for ix in range(len(pop)):
    # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

#Operation de croisement
def crossover(p1, p2, r_cross):
 # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
 # check for recombination
    if random.random() < r_cross:
 # select crossover point that is not on the end of the string
        pt = random.randint(1, len(p1)-2)
 # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

#Operateur de mutation
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
 # check for a mutation
        if random.random() < r_mut:
 # flip the bit
            bitstring[i] = 1 - bitstring[i]



# L'algorithme génétique

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, population):
 # initial population of random bitstring
    pop = population
 # keep track of best solution
    best, best_eval = 0, objective(pop[0] )
    Scores=[]
 # enumerate generations
    for gen in range(n_iter):
 # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        Scores.append(scores)
 # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                 best, best_eval = pop[i], scores[i]
                 #print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
 # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
 # create the next generation
    children = list()
    for i in range(0, n_pop-1, 2):
 # get selected parents in pairs
         p1, p2 = selected[i], selected[i+1]
 # crossover and mutation
         for c in crossover(p1, p2, r_cross):
 # mutation
                mutation(c, r_mut)
 # store for next generation
                children.append(c)
 # replace population
         pop = children
    print("Scores=", Scores)
    return [best, best_eval]

#print(genetic_algorithm(objective,0,260,len(population),0.8,0.2))

        