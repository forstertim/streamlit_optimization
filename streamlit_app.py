'''
Installing GLPK: conda install -c conda-forge glpk
Installing IPOPT: conda install -c "conda-forge/label/cf202003" ipopt
'''
import numpy as np
import streamlit as st
import pyomo.environ as pyo
import matplotlib.pyplot as plt


# STREAMLIT INPUTS
####################################################################
st.title("Let's solve an optimization problem!")

# Drop down menu for objective function
chosen_objective = st.selectbox("Choose objective function:", ("Linear", "Nonlinear"))

if chosen_objective == "Nonlinear":

    st.info(f"You chose to minimize a {chosen_objective.upper()} function! \n\nThe problem we are trying to solve is the following: \n\n$\min_{{x_1,x_2}} \quad c_1x_1^2 + c_2x_2 $ \n\n s.t. $\qquad a_1x_1 + a_2x_2 \geq b$ \n\n $\qquad \quad x_i \geq 0, \quad i=1,2$")
    st.error("Curently, a nonlinear solver is not implemented yet! In the meantime, you can generate some random emojis below 🥰")
    import random
    def random_emoji():
        st.session_state.emoji = random.choice(emojis)
    if "emoji" not in st.session_state:
        st.session_state.emoji = "👈"
    emojis = ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼"]
    st.button(f"Click Me {st.session_state.emoji}", on_click=random_emoji)

else:

    # Display problem
    st.info(f"You chose to minimize a {chosen_objective.upper()} function! \n\nThe problem we are trying to solve is the following: \
            \n\n$\min_{{x_1,x_2}} \quad c_1x_1 + c_2x_2 $ \
            \n\n subject to: \
            \n\n $\qquad f_1: \quad a_1x_1 + a_2x_2 \geq b_1$ \
            \n\n $\qquad f_2: \quad x_2 \geq b_2$ \
            \n\n $\qquad f_3: \quad x_i \geq 0, \quad i=1,2$")

    # Slider for coefficients of objective function
    st.header("Choose coefficients for the problem above:")

    obj_coef_c1 = st.slider(r"Choose objective function coefficient $c_1$ (influences the slope of the contour lines):", 
                        min_value=1.0, max_value=12.0, value=2.0, step=0.1)
    obj_coef_c2 = st.slider(r"Choose objective function coefficient $c_2$ (influences the slope of the contour lines):", 
                        min_value=1.0, max_value=12.0, value=3.0, step=0.1)

    # Slider for coefficients of constraint function
    constr_coef_a1 = st.slider(r"Choose constraint function coefficient $a_1$ (influences the slope of the constraint line $f_1$):", 
                        min_value=1.0, max_value=12.0, value=3.0, step=0.1)
    constr_coef_a2 = st.slider(r"Choose constraint function coefficient $a_2$ (influences the slope and y-intercept of the constraint line $f_1$):", 
                        min_value=1.0, max_value=8.0, value=4.0, step=0.1)

    # Slider for b value of constraint function
    constr_coef_b1 = st.slider(r"Choose constraint value $b_1$ (influences the y-intercept of the constraint line $f_1$):", 
                        min_value=3.0, max_value=10.0, value=8.0, step=0.1)
    constr_coef_b2 = st.slider(r"Choose constraint value $b_2$ (influences the y-intercept of the constraint line $f_2$):", 
                        min_value=0.0, max_value=4.5, value=1.0, step=0.1)


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
    model.Constraint2 = pyo.Constraint(expr = model.decisionvariable[2] >= constr_coef_b2)

    # Get lambda equations for plotting later on
    constraint_func_f1 = lambda x_1_: (constr_coef_b1 - constr_coef_a1*x_1_)/constr_coef_a2
    def constraint_func_f2(length):
        return np.ones(length)*constr_coef_b2

    # Choose solver and solve model
    # solver = pyo.SolverFactory('ipopt')
    # solver.solve(model, mip_solver='glpk', nlp_solver='ipopt') 
    solver = pyo.SolverFactory('glpk')
    solver.solve(model, tee=True) 


    # STREAMLIT OUTPUTS
    ####################################################################
    st.title("Optimal solution")
    st.success(f'Since the problem is a minimization problem, the optimization direction goes towards the lower values of the contour lines. \n\nThe optimal solution is given by the following: \
               \n\n $F^*= {model.obj():.2f} \quad$ with $\quad$ $(x_1^*, x_2^*)=({model.decisionvariable[1]():.2f}, {model.decisionvariable[2]():.2f})$')


    # STREAMLIT VISUALIZATION
    ####################################################################
    # Plot settings
    eps = 1e-3
    x1min = 0
    x2min = 0
    x1max = 5
    x2max = 5
    gridpoints = 100
    x1_ = np.linspace(x1min,x1max,gridpoints)
    x2_ = np.linspace(x2min,x2max,gridpoints)
    X1, X2 = np.meshgrid(x1_, x2_)
    F = obj_func(X1, X2)

    # Get figure
    fig,ax = plt.subplots()
    # Plot objective controur lines
    CS=ax.contour(X1, X2, F, levels=10, linestyles='solid', linewidths=1)
    ax.clabel(CS, CS.levels, inline=True, fontsize=8)
    # Plot constraint
    ax.plot(x1_, constraint_func_f1(x1_), label=f'Constraint $f_1$', marker='', linestyle='--', color='k')
    ax.plot(x1_, constraint_func_f2(gridpoints), label=f'Constraint $f_2$', marker='', linestyle=':', color='k')
    # Plot feasible region
    topxaxis_ = np.ones(len(x1_))*x2max
    ax.fill_between(x1_, topxaxis_, constraint_func_f1(x1_), color='C0', alpha=0.2, label='Feasible region')
    ax.fill_between(x1_, constraint_func_f1(x1_), constraint_func_f2(gridpoints), color='white', alpha=1)
    # Plot optimal solution
    ax.scatter(model.decisionvariable[1](), model.decisionvariable[2](), label=f'Optimal solution', marker='o', color='r', s=80)
    # Plot settings and legend
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([x1min,x1max])
    plt.ylim([x2min,x2max])
    plt.legend()
    st.pyplot(fig)