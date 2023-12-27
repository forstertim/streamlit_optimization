'''
Installing GLPK: conda install -c conda-forge glpk
Installing IPOPT: conda install -c "conda-forge/label/cf202003" ipopt
'''
import numpy as np
import streamlit as st
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# FIND IPOPT PATH (TEMPORARY CODE)
####################################################################
import subprocess
ipopt_path = subprocess.check_output(['which', 'ipopt']).decode('utf-8').strip()
str.write(f"Ipopt executable path: {ipopt_path}")

# STREAMLIT INPUTS
####################################################################
st.title("Choose properties")

# Drop down menu for objective function
chosen_objective = st.selectbox("Choose objective function:", ("Linear", "Nonlinear"))
st.write(f"You chose to minimize a {chosen_objective.upper()} function!")

# Slider for coefficients of objective function
obj_coef_c1 = st.slider(r"Choose objective function coefficient $c_1$:", 
                     min_value=1.0, max_value=12.0, value=2.0, step=0.1)
obj_coef_c2 = st.slider(r"Choose objective function coefficient $c_2$:", 
                     min_value=1.0, max_value=12.0, value=3.0, step=0.1)

# Slider for coefficients of constraint function
constr_coef_a1 = st.slider(r"Choose constraint function coefficient $a_1$:", 
                     min_value=1.0, max_value=12.0, value=3.0, step=0.1)
constr_coef_a2 = st.slider(r"Choose constraint function coefficient $a_2$:", 
                     min_value=1.0, max_value=8.0, value=4.0, step=0.1)

# Slider for b value of constraint function
constr_coef_b1 = st.slider(r"Choose constraint value $b$:", 
                     min_value=3.0, max_value=10.0, value=5.0, step=0.1)


# PYOMO MODEL
####################################################################
# Create a concrete model
model = pyo.ConcreteModel()

# Create variables
model.decisionvariable = pyo.Var([1,2], domain=pyo.NonNegativeReals)

# Create objective (linear or non-linear)
if chosen_objective == "Linear":
    model.obj = pyo.Objective(expr = obj_coef_c1*model.decisionvariable[1] + obj_coef_c2*model.decisionvariable[2], sense=pyo.minimize)
    obj_func = lambda x_1_, x_2_: obj_coef_c1*x_1_ + obj_coef_c2*x_2_
elif chosen_objective == "Nonlinear":
    model.obj = pyo.Objective(expr = obj_coef_c1*model.decisionvariable[1]**2 + obj_coef_c2*model.decisionvariable[2], sense=pyo.minimize)
    obj_func = lambda x_1_, x_2_: obj_coef_c1*x_1_**2 + obj_coef_c2*x_2_

# Create constraint
model.Constraint1 = pyo.Constraint(expr = constr_coef_a1*model.decisionvariable[1] + constr_coef_a2*model.decisionvariable[2] >= constr_coef_b1)

# Get lambda equations for plotting later on
constraint_func = lambda x_1_: (constr_coef_b1 - constr_coef_a1*x_1_)/constr_coef_a2

# Choose solver and solve model
# solver = pyo.SolverFactory('ipopt')
# solver.solve(model, mip_solver='glpk', nlp_solver='ipopt') 
solver = pyo.SolverFactory('glpk')
solver.solve(model, tee=True) 


# STREAMLIT OUTPUTS
####################################################################
st.title("Optimal solution")
st.write(f'$F^*=${model.obj():.2f} \t with ($x_1^*$, $x_2^*$)=({model.decisionvariable[1]():.2f}, {model.decisionvariable[2]():.2f})')


# STREAMLIT VISUALIZATION
####################################################################
st.title("Visualization")
# Plot settings
x1max = 5
x2max = 5
gridpoints = 100
x1_ = np.linspace(0,x1max,gridpoints)
x2_ = np.linspace(0,x2max,gridpoints)
X1, X2 = np.meshgrid(x1_, x2_)
F = obj_func(X1, X2)

# Get figure
fig,ax = plt.subplots()
# Plot objective controur lines
CS=ax.contour(X1, X2, F, levels=10, linestyles='dashed', linewidths=1)
ax.clabel(CS, CS.levels, inline=True, fontsize=8)
# Plot constraint
ax.plot(x1_, constraint_func(x1_), label=f'Constraint 1', marker='', linestyle='-', color='k')
# Plot optimal solution
ax.scatter(model.decisionvariable[1](), model.decisionvariable[2](), label=f'Optimal solution', marker='o', color='r')
# Plot settings and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([0,x1max])
plt.ylim([0,x2max])
plt.legend()
st.pyplot(fig)