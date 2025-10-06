"""
Author: Ratan Lal
Date : 4 January, 2021
"""
import numpy as np
from scipy.optimize import linprog
from gurobipy import *


class StarSet:
	"""
	Class representing starset
	Star set defined by 
	x = c+a[1]*v[1] + a[2]*v[2]+ a[n]*v[n]
	  = V*b, where V = [c v[1], v[2], ..., v[n]]	
		and b =[1, a[1], a[2], ..., a[n]]^{T}	
	C*a <=d, constraints on a[i]
	Attributes:
		matBasisV (Matrix) : basis matrix
		matConstraintC (Matrix) : constraint matrix
		cvecConstraintd (column vector) : constraint vector
		intDim (int) : dimension of the set
		intNumVar (int) : number of variables in the constraints
		cvecPredicateLb (column vector) : lower bound vector of predicate variable
		cvecPredicateUb (column vector) : upper bound vector of predicate variable
		cvecStateLb (column vector) : lower bound vector of state variable
		cvecStateUb (column vector) : upper bound vector of state variable
		zonotopeZ (zonotope) : an outer zonotope covering this star, used for reachability of logsig and tansig networks	
	"""
	
	def __init__(self, matBasisV=None, matConstraintC=None, cvecConstraintd=None, cvecPredicateLb=None, cvecPredicateUb=None,cvecStateLb=None, cvecStateUb=None, zonotopeZ=None):
		#two dimensional numpy array
		self.matBasisV = matBasisV
		#two dimensional numpy array
		self.matConstraintC = matConstraintC
		#two dimensional numpy array with one column
		self.cvecConstraintd = cvecConstraintd
		# derived properties
		self.intDim = matBasisV.shape[0]
		self.intNumVar = matConstraintC.shape[1]
		#two dimensional numpy array with one column
		if cvecPredicateLb==None:
			self.cvecPredicateLb = np.array([[0] for i in range(self.intNumVar)], dtype='f') 
			self.cvecPredicateLb = np.array([[0] for i in range(self.intNumVar)], dtype='f') 
		else:
			self.cvecPredicateLb = cvecPredicateLb
			self.cvecPredicateUb = cvecPredicateUb
		#two dimensional numpy array with one column
		if (cvecStateLb==None):
			self.cvecStateLb = np.array([[0] for i in range(self.intDim)], dtype='f') 
			self.cvecStateUb = np.array([[0] for i in range(self.intDim)], dtype='f') 
		else:
			self.cvecStateLb = cvecStateLb
			self.cvecStateUb = cvecStateUb
		
		#an instance of Zonotope
		self.zonotopeZ = zonotopeZ

	def isEmpty(self):
		"""
		Check whether starset is empty
		Returns:
			status (bool): True if empty or False
		"""
		try:
			#linear programming
			c = [0 for i in range(self.intNumVar)]
			# matrix
			A = self.matConstraintC
	
			#vector
			b = [self.cvecConstraintd[i,0] for i in range(self.cvecConstraintd.shape[0])]
			
			#bounds
			bound = [(None, None) for i in range(self.intNumVar)]
	
			#feasibility
			res = linprog(c, A_ub=A, b_ub=b, bounds=bound)
			#print(res)
			return not(res.success)
		except:
			return True
	


	def quadrantPartition(self):
		"""
		Paritioning of a star set
		Return:
			listStarSet : list of instances of StarSet
			listoflistSign: list of list of sign for quadrant
		"""
		listStarSet = [(self,'')]
		for i in range(self.intDim):
			tempList = []
			for S in listStarSet:
				tempS = S[0].intersectPositiveHalfSpace(i)
				if not(tempS.isEmpty()):
					tempList.append((tempS,S[1]+'+ '))
				tempS = S[0].intersectNegativeHalfSpace(i)
				if not(tempS.isEmpty()):
					tempList.append((tempS,S[1]+'- '))

			listStarSet = tempList
		
		listS = []
		listSign = []
		for i in range(len(listStarSet)):
			listS.append(listStarSet[i][0])
			temp = listStarSet[i][1][:-1].split(' ')
			listSign.append(temp)

		return listS, listSign
		
	def affineMapWInput(self, matW=None):
		"""
		affine transformation of a star set
			S = matWx
		
		Args:
			matW (matrix) : Mapping matrix
			
		Returns:
			StarSetS (StarSet) : new star set
		"""
		# matrix multiplication
		
		matbasisnewV = np.array(np.matmul(matW, self.matBasisV))
		
		#create new star set
		objStarSet = StarSet(matbasisnewV, self.matConstraintC, self.cvecConstraintd)
		return objStarSet
		
	def affineMap(self, matW=None, cvecb = None):
		"""
		affine transformation of a star set
			S = matWx+cvecb
		
		Args:
			matW (matrix) : Mapping matrix
			cvecb (column vector) : mapping vector
			
		Returns:
			StarSetS (StarSet) : new star set
		"""
		# matrix multiplication
		matbasisnewV = np.array(np.matmul(matW, self.matBasisV))
		# sum of column vectors c' = c' + b
		matbasisnewV[:,0] = matbasisnewV[:,0] + cvecb[:,0]
		#create new star set
		objStarSet = StarSet(matbasisnewV, self.matConstraintC, self.cvecConstraintd)
		return objStarSet

	def minkowskiSum(self, objStarSet=None):
		"""
		Monkowski sum of two star sets
		input:
			objStarset (Starset): an instance of Starset

		output:
			newStarset (Starset): a new Starset
		"""
		#horizontal concatenation
		matBasisV = np.concatenate((self.matBasisV, objStarSet.matBasisV[:,1:]), axis=1)
		#addition of centers
		matBasisV[:,0] = self.matBasisV[:,0] + objStarSet.matBasisV[:,0] 
		#block diagonal of self.C and objStarset.C
		r1 = np.shape(self.matConstraintC)[0]
		c1 =  np.shape(self.matConstraintC)[1]
		r2 = np.shape(objStarSet.matConstraintC)[0]
		c2 = np.shape(objStarSet.matConstraintC)[1]
		matConstraintC = np.block([[self.matConstraintC, np.zeros((r1,c2))],[np.zeros((r2,c1)), objStarSet.matConstraintC]])
		#cvecConstraintd
		cvecConstraintd = np.concatenate((self.cvecConstraintd, objStarSet.cvecConstraintd), axis=0)
		
		objSumStarSet = StarSet(matBasisV, matConstraintC, cvecConstraintd)
		
		return objSumStarSet		
	
		
	def convexHull(self, objStarSet=None):
		"""
		Compute convex hull of star sets
		"""
		c1 = self.matBasisV[:,0]
		c2 = objStarSet.matBasisV[:,0]
		V1 = self.matBasisV
		V1[:,0] = c1 - c2
	
		V2 = objStarSet.matBasisV[:,1:]
		newV = np.concatenate((V1, -V2), axis=1)
		r1 = np.shape(self.matConstraintC)[0]
		c1 =  np.shape(self.matConstraintC)[1]
		r2 = np.shape(objStarSet.matConstraintC)[0]
		c2 = np.shape(objStarSet.matConstraintC)[1]
		newC = np.block([[self.matConstraintC, np.zeros((r1,c2))],[np.zeros((r2,c1)), objStarSet.matConstraintC]])
		newd = np.concatenate((self.cvecConstraintd, objStarSet.cvecConstraintd), axis=0)
		S2 = StarSet(newV,newC, newd)
		S = objStarSet.minkowskiSum(S2)
		return S	

	def intersectHalfSpace(self, matH=None, cvecg=None):
		"""
		intersection between star set and half space Hx<=g (full dimension)
		
		Args:
			matH (Matrix) : Half space matrix
			cvecg (Column vector) : half space vector
			
		Returns:
			objStarSet (Starset) : new star set
		"""
		#matrix multiplication of H and vertices of star set
		C1 = np.matmul(matH, self.matBasisV[:,1:])
		# substraction of two column vectors and multipication of a matrix and the center column vector
		
		cvecg[:,0] = cvecg[:,0] - np.matmul(matH, self.matBasisV[:,0])
		
		#vertical concatenation
		matConstraintnewC = np.concatenate((self.matConstraintC, C1), axis=0)
		#vertical concatenation
		cvecConstraintnewd = np.concatenate((self.cvecConstraintd, cvecg), axis=0)
		#create new star set
		objStarSet = StarSet(self.matBasisV, matConstraintnewC, cvecConstraintnewd)
		return objStarSet
		
		
	def intersectPositiveHalfSpace(self, intIndex=None):
		"""
		intersect with positive half spaces x[intIndex] >=0
		
		Args:
			intIndex (int) : index of a star set
		
		Returns:
			objStarSet (StarSet) : new star set 
		"""
		#vertical conncatenation (array dimension should be matched, for instance 2D array, 3D array)
		matConstraintnewC = np.concatenate((self.matConstraintC, np.array([-self.matBasisV[intIndex,1:]])), axis=0)
		#vertical conncatenation
		cvecConstraintnewd = np.concatenate((self.cvecConstraintd, np.array([[self.matBasisV[intIndex,0]]])), axis=0)
		#create star set
		objStarSet = StarSet(self.matBasisV, matConstraintnewC, cvecConstraintnewd)
		return objStarSet
		
	def intersectNegativeHalfSpace(self, intIndex=None):
		"""
		intersect with Negative half spaces x[intIndex] <=0
		
		Args:
			intIndex (int) : index of a star set
		
		Returns:
			objStarSet (StarSet) : new star set 
		"""
		#vertical conncatenation (array dimension should be matched, for instance 2D array, 3D array)
		matConstraintnewC = np.concatenate((self.matConstraintC, np.array([self.matBasisV[intIndex,1:]])), axis=0)
		#vertical conncatenation
		cvecConstraintnewd = np.concatenate((self.cvecConstraintd, np.array([[-self.matBasisV[intIndex,0]]])), axis=0)
		#create star set
		objStarSet = StarSet(self.matBasisV, matConstraintnewC, cvecConstraintnewd)
		return objStarSet
	
	def createConstraints(self):
		"""
		Create all constraints corresponding to the star set
		
		Returns:
			m (Model) : an instance of gurobi model
		"""
		#create variables
		stateVars = ['x_'+str(i) for i in range(self.intDim)]
		predicateVars = ['alpha_'+str(j) for j in range(self.intNumVar)]	
		#linear programming
		m = Model()	
		#create variables x = (x1, x2,..., xn)
		for i in range(self.intDim):
			stateVars[i] = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=stateVars[i])
		#create predicate variables a = (a1, a2, ..., an)
		for j in range(self.intNumVar):
			predicateVars[j] = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=predicateVars[j])
		#create constraints for states x = c + sum (ai vi)
		for i in range(self.intDim):
			m.addConstr(stateVars[i] == self.matBasisV[i,0] + quicksum(predicateVars[j]*self.matBasisV[i,j+1] for j in range(self.intNumVar)))
		#create constraints for predicates C*a <= d
		for i in range(len(self.cvecConstraintd[:,0])):
			m.addConstr(quicksum(self.matConstraintC[i,j]*predicateVars[j] for j in range(self.intNumVar)) <= self.cvecConstraintd[i,0])
		#update the model
		m.update()
		return m, stateVars


	def createConstraintsbyIndex(self, intIndex=None):
		"""
		Create constraints for variable at index intIndex
		Returns:
			m (Model): an instance of gurobi model
		"""
		#create variables
		stateVars = ['x_'+str(intIndex)]
		predicateVars = ['alpha_'+str(j) for j in range(self.intNumVar)]	
		#linear programming
		m = Model()	
		#create variables x = x_intIndex
		stateVars[0] = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=stateVars[0])
		#create predicate variables a = (a1, a2, ..., an)
		for j in range(self.intNumVar):
			predicateVars[j] = m.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name=predicateVars[j])
		#create constraints for states x = c + sum (ai vi)
		m.addConstr(stateVars[0] == self.matBasisV[intIndex,0] + quicksum(predicateVars[j]*self.matBasisV[intIndex,j+1] for j in range(self.intNumVar)))
		#create constraints for predicates C*a <= d
		for i in range(len(self.cvecConstraintd[:,0])):
			m.addConstr(quicksum(self.matConstraintC[i,j]*predicateVars[j] for j in range(self.intNumVar)) <= self.cvecConstraintd[i,0])
		#update the model
		m.update()
		return m, stateVars

		
		
	def updateStateLb(self, stateVars):
		"""
		Extract lower values from the model.sol and update the column vector for stateLb
		"""
		f = open('model.sol', 'r')
		contents = f.readlines()
		
		self.cvecStateLb = np.array([[0] for i in range(self.intDim)], dtype='f') 
		for i in range(self.intDim):
			line = contents[i+1]
			self.cvecStateLb[i,0] = float(line.split(' ')[1].strip('\n'))

	def updateStateLbbyIndex(self, intIndex=None):
		"""
		Extract lower values from the model.sol and update the column vector at indeIndex for stateLb
		"""
		f = open('model.sol', 'r')
		contents = f.readlines()
		value = float(contents[1].split(' ')[1].strip('\n'))
		self.cvecStateLb[intIndex,0] = value

		return value
			
	def updateStateUb(self, stateVars):
		"""
		Extract upper values from the model.sol and update the column vector for stateUb
		"""
		f = open('model.sol', 'r')
		contents = f.readlines()
		
		self.cvecStateUb = np.array([[0] for i in range(self.intDim)],dtype='f') 
		for i in range(self.intDim):
			line = contents[i+1]
			self.cvecStateUb[i,0] = float(line.split(' ')[1].strip('\n'))

	def updateStateUbbyIndex(self, intIndex=None):
		"""
		Extract upper values from the model.sol and update the column vector at intIndex for stateUb
		"""
		f = open('model.sol', 'r')
		contents = f.readlines()
		value = float(contents[1].split(' ')[1].strip('\n'))
		self.cvecStateUb[intIndex,0] = value
		return value
			
	
	def getRangebyIndex(self, intIndex=None):
		"""
		Compute range of starset with respect to intIndex dimension
		Args:
			indeIndex (index): index of state variable

		Returns:
			(lB, uB): lower and upper bound
		"""			
		
		#linear programming
		c = [0 for i in range(self.intDim + self.intNumVar)]
		c[intIndex] = 1
		# matrix
		r1 = self.matConstraintC.shape[0]
		#InEquality array
		Aub = np.array(np.concatenate((np.zeros((r1,self.intDim)),self.matConstraintC), axis=1))
		bub = np.array([self.cvecConstraintd[i,0] for i in range(self.cvecConstraintd.shape[0])], dtype='f')
		Aeq = np.array([[0 for i in range(self.intDim + self.intNumVar)]], dtype='f')
		Aeq[0,intIndex] = 1
		
		for j in range(self.intNumVar):
			Aeq[0,self.intDim+j] = -self.matBasisV[intIndex,j+1]
		beq = np.array([self.matBasisV[intIndex,0]], dtype='f')

		#bounds
		bound = [(None, None) for i in range(self.intDim+ self.intNumVar)]
		
		#minimization
		res = linprog(c, A_ub=Aub, b_ub=bub, A_eq = Aeq, b_eq=beq, bounds=bound)
		Lb= res.x[intIndex]
		#print(Lb)
		#maximization
		
		c[intIndex] = -1
		res = linprog(c, A_ub=Aub, b_ub=bub, A_eq = Aeq, b_eq=beq, bounds=bound)
		Ub= res.x[intIndex]
		#print(Ub)
		
		'''
		#by Gurobi
		#add all constraints of the starset to a gurobi model
		m, stateVars = self.createConstraintsbyIndex(intIndex)
		#compute lower range 
		m.setObjective(stateVars[0], GRB.MINIMIZE)
		m.update()	
		m.optimize()
		m.write('./model.sol')
		Lb = self.updateStateLbbyIndex(intIndex)
		m.setObjective(stateVars[0], GRB.MAXIMIZE)
		m.update()	
		m.optimize()
		m.write('./model.sol')
		Ub = self.updateStateUbbyIndex(intIndex)
		'''
		return (Lb, Ub)


		
	def getRange(self):
		"""
		Compute the range of a star set
		
		Returns:
			cvecLb (column vector) : lower bound on the star set
			cvecUb (column vector) : upper bound on the star set
		"""
		#add all constraints of the starset to a gurobi model
		m, stateVars = self.createConstraints()
		#compute lower range 
		m.setObjective(quicksum(stateVars[i] for i in range(self.intDim)), GRB.MINIMIZE)
		m.update()	
		m.optimize()
		#for infeasible constraints
		#m.computeIIS()
		#m.write('./model.ilp')
		#for linear constraints
		#m.write('./model.lp')
		#for model
		m.write('./model.sol')
		self.updateStateLb(stateVars)

		#compute upper range
		m.setObjective(quicksum(stateVars[i] for i in range(self.intDim)), GRB.MAXIMIZE)
		m.update()	
		m.optimize()
		m.write('./model.sol')
		self.updateStateUb(stateVars)

	def checkSafety(self, listA=None, listb=None):
		"""
		Checking safety 
		Args:
			A (matrix) : a matrix of size m xn
			b (column vector) : column bector of size mx1

		"""
		for k in range(len(listA)):
			m, stateVars = self.createConstraints()
			A = listA[k]
			b = listb[k]
			(Rows, Cols) = A.shape
			#print(Rows)
			#print(Cols)
			for i in range(Rows):
				m.addConstr(quicksum(A[i][j]*stateVars[j] for j in range(Cols))<= b[i,0])
			m.update()
			m.optimize()	
			if m.STATUS==2:
				return True
			if m.STATUS==3:
				continue
		return False
		
	
		

		
			
		
		
		
	
	
		

			
		
		

