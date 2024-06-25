'''
Installing GLPK: conda install -c conda-forge glpk
Installing IPOPT: conda install -c "conda-forge/label/cf202003" ipopt
'''
import numpy as np
import streamlit as st
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from gekko import GEKKO


# STREAMLIT INPUTS
####################################################################
# st.set_page_config(layout="wide")
st.title("Let's solve an optimization problem!")

# Drop down menu for objective function
chosen_objective = st.selectbox("Choose objective function:", ("Linear", "Nonlinear"))

if chosen_objective == "Nonlinear":

    # Display problem
    st.info(f"You chose to minimize a {chosen_objective.upper()} function! We will solve this with GEKKO (and IPOPT). \n\nThe problem we are trying to solve is the following: \
            \n\n$\min_{{x_1,x_2}} \quad c_1x_1^2 + c_2x_2 $ \
            \n\n subject to: \
            \n\n $\qquad f_1: \quad a_1x_1 + a_2x_2 \geq b_1$ \
            \n\n $\qquad f_2: \quad x_2 \geq b_2$ \
            \n\n $\qquad f_3: \quad x_i \geq 0, \quad i=1,2$")
    st.warning("The nonlinear problem solved with IPOPT might take a bit more time than the linear objective with GLPK, so be patient 🥰")
    st.warning('It might be more obvious to see where the contour lines intersect the constraints if you change the number of contour lines until you see an intersection 😊')

else:

    # Display problem
    st.info(f"You chose to minimize a {chosen_objective.upper()} function! We will solve this with PYOMO (and GLPK).\n\nThe problem we are trying to solve is the following: \
            \n\n$\min_{{x_1,x_2}} \quad c_1x_1 + c_2x_2 $ \
            \n\n subject to: \
            \n\n $\qquad f_1: \quad a_1x_1 + a_2x_2 \geq b_1$ \
            \n\n $\qquad f_2: \quad x_2 \geq b_2$ \
            \n\n $\qquad f_3: \quad x_i \geq 0, \quad i=1,2$")
    
# If nonlinear, decide how many contour lines should be displayed
if chosen_objective == "Nonlinear":
    num_contour_lines = st.slider(r"Choose number of contour lines:", 
                        min_value=8.0, max_value=30.0, value=10.0, step=2.0)
else:
    num_contour_lines = 10.0

# Create columns
col1, col2, col3 = st.columns(3)

# Slider for coefficients of objective function
with col1:
    st.header("Objective coefficients:")
    obj_coef_c1 = st.slider(r"Choose $c_1$:", 
                        min_value=1.0, max_value=12.0, value=2.0, step=0.1)
    obj_coef_c2 = st.slider(r"Choose $c_2$:", 
                        min_value=1.0, max_value=12.0, value=3.0, step=0.1)

# Slider for coefficients of constraint function
with col2:
    st.header('Constraint coefficients:')
    constr_coef_a1 = st.slider(r"Choose $a_1$:", 
                        min_value=1.0, max_value=12.0, value=3.0, step=0.1)
    constr_coef_a2 = st.slider(r"Choose $a_2$:", 
                        min_value=1.0, max_value=8.0, value=4.0, step=0.1)

# Slider for b value of constraint function
with col3:
    st.header('Constraint values:')
    constr_coef_b1 = st.slider(r"Choose $b_1$:", 
                        min_value=3.0, max_value=10.0, value=8.0, step=0.1)
    constr_coef_b2 = st.slider(r"Choose $b_2$:", 
                        min_value=0.0, max_value=4.5, value=1.0, step=0.1)

# Variable fixed settings
eps = 1e-3
x1min = 0
x2min = 0
x1max = 5
x2max = 5

# PYOMO MODEL FOR LINEAR MODEL
####################################################################
if chosen_objective == 'Linear':
    # Create a concrete model
    model = pyo.ConcreteModel()
    # Create variables
    model.decisionvariable = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    # Create objective 
    model.obj = pyo.Objective(expr = obj_coef_c1*model.decisionvariable[1] + obj_coef_c2*model.decisionvariable[2], sense=pyo.minimize)
    # Create constraint
    model.Constraint1 = pyo.Constraint(expr = constr_coef_a1*model.decisionvariable[1] + constr_coef_a2*model.decisionvariable[2] >= constr_coef_b1)
    model.Constraint2 = pyo.Constraint(expr = model.decisionvariable[2] >= constr_coef_b2)
    # Solve problem
    solver = pyo.SolverFactory('glpk')
    solver.solve(model, tee=True) 
    # Get optimal solution
    x1opt = model.decisionvariable[1]()
    x2opt = model.decisionvariable[2]()
    Fopt = model.obj()
    # To draw objective function 
    obj_func = lambda x_1_, x_2_: obj_coef_c1*x_1_ + obj_coef_c2*x_2_
        
        
# GEKKO MODEL FOR NONLINEAR MODEL
####################################################################
elif chosen_objective == "Nonlinear":
    # Create model
    model = GEKKO(remote=False)
    # Create variables 
    x1 = model.Var(lb=x1min, ub=x1max, value=x1max)
    x2 = model.Var(lb=x2min, ub=x2max, value=x2max)
    # Create constraints
    model.Equation(constr_coef_a1*x1 + constr_coef_a2*x2 >= constr_coef_b1)
    model.Equation(x2 >= constr_coef_b2)
    # Create objective
    model.Minimize(obj_coef_c1*x1**2 + obj_coef_c2*x2)
    # Solve problem
    model.solve()
    # Get optimal solution
    x1opt = x1.value[0]
    x2opt = x2.value[0]
    Fopt = model.options.OBJFCNVAL
    # To draw objective function 
    obj_func = lambda x_1_, x_2_: obj_coef_c1*x_1_**2 + obj_coef_c2*x_2_
        
        
# GET FUNCTIONS FOR PLOTTING
####################################################################
constraint_func_f1 = lambda x_1_: (constr_coef_b1 - constr_coef_a1*x_1_)/constr_coef_a2
def constraint_func_f2(length):
    return np.ones(length)*constr_coef_b2


# STREAMLIT VISUALIZATION AND SOLUTION RETURNS
####################################################################
# Columns for visualizations and solution retursn    
dispcol1, dispcol2 = st.columns(2)

# Plot settings
gridpoints = 50
x1_ = np.linspace(x1min,x1max,gridpoints)
x2_ = np.linspace(x2min,x2max,gridpoints)
X1, X2 = np.meshgrid(x1_, x2_)
F = obj_func(X1, X2)

# Get figure
fig,ax = plt.subplots()
# Plot objective controur lines
CS=ax.contour(X1, X2, F, levels=int(num_contour_lines), linestyles='solid', linewidths=1)
ax.clabel(CS, CS.levels, inline=True, fontsize=8)
# Plot constraint
ax.plot(x1_, constraint_func_f1(x1_), label=f'Constraint $f_1$', marker='', linestyle='--', color='k')
ax.plot(x1_, constraint_func_f2(gridpoints), label=f'Constraint $f_2$', marker='', linestyle=':', color='k')
# Plot feasible region
topxaxis_ = np.ones(len(x1_))*x2max
ax.fill_between(x1_, topxaxis_, constraint_func_f1(x1_), color='C0', alpha=0.2, label='Feasible region')
ax.fill_between(x1_, constraint_func_f1(x1_), constraint_func_f2(gridpoints), color='white', alpha=1)
# Plot optimal solution
ax.scatter(x1opt, x2opt, label=f'Optimal solution', marker='o', color='r', s=80)
# Plot settings and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([x1min,x1max])
plt.ylim([x2min,x2max])
plt.legend()
with dispcol1:
    st.pyplot(fig)

with dispcol2:
    st.title("Optimal solution")
    st.success(f'Since the problem is a minimization problem, the optimization direction goes towards the lower values of the contour lines. \n\nThe optimal solution is given by the following: \
                \n\n $F^*= {Fopt:.2f} \quad$ with $\quad$ $(x_1^*, x_2^*)=({x1opt:.2f}, {x2opt:.2f})$')