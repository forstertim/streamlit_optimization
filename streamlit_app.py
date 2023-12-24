'''
Installing GLPK: conda install -c conda-forge glpk
Installing IPOPT: conda install -c "conda-forge/label/cf202003" ipopt
'''

import streamlit as st
import pyomo.environ as pyo

# STREAMLIT INPUTS
####################################################################
st.title("Choose properties")

# Slider for coefficients of objective function
obj_coef_c1 = st.slider(r"Choose objective function coefficient $c_1$:", 
                     min_value=1.0, max_value=12.0, value=2.0, step=0.1)
obj_coef_c2 = st.slider(r"Choose objective function coefficient $c_2$:", 
                     min_value=1.0, max_value=12.0, value=3.0, step=0.1)

# Slider for coefficients of constraint function
constr_coef_a1 = st.slider(r"Choose constraint function coefficient $a_1$:", 
                     min_value=1.0, max_value=12.0, value=3.0, step=0.1)
constr_coef_a2 = st.slider(r"Choose constraint function coefficient $a_2$:", 
                     min_value=1.0, max_value=12.0, value=4.0, step=0.1)

# Slider for b value of constraint function
constr_coef_b = st.slider(r"Choose constraint value $b$:", 
                     min_value=0.0, max_value=5.0, value=1.0, step=0.1)


# PYOMO MODEL
####################################################################
# Create a concrete model
model = pyo.ConcreteModel()

# Create variables, objective function and constraints
model.decisionvariable = pyo.Var([1,2], domain=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr = obj_coef_c1*model.decisionvariable[1] + obj_coef_c2*model.decisionvariable[2])
model.Constraint1 = pyo.Constraint(expr = constr_coef_a1*model.decisionvariable[1] + constr_coef_a2*model.decisionvariable[2] >= constr_coef_b)

# Choose solver and solve model
# solver = pyo.SolverFactory('mindtpy')
# solver.solve(model, mip_solver='glpk', nlp_solver='ipopt') 
solver = pyo.SolverFactory('glpk')
solver.solve(model) 


# STREAMLIT OUTPUTS
####################################################################
st.title("Optimal solution")

st.header("Decision variables")
st.write(model.decisionvariable[1]())
st.write(model.decisionvariable[2]())

st.header("Objective function")
st.write(model.obj())