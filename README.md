# Mathematical Optimization with Streamlit :rocket:

Let's run an optimization problem on Streamlit :heart:

In the top of this simple app, one can choose the type of objective function we would like to minimize, which is either a `linear` one, or a `nonlinear` one. 

Since this is a very simple example, we stick to the following objective functions:

- `linear`: $F(x) = c_1x_1 + c_2x_2$
- `nonlinear`: $F(x) = c_1x_1^2 + c_2x_2$

The constraints are the following (equal in both cases):

- $f_1: \quad a_1x_1 + a_2x_2 \geq b_1$
- $f_2: \quad x_2 \geq b_2$
- $f_3: \quad x_i \geq 0, \quad i=1,2$

The app is stored on the publicly available Streamlit cloud [here :partly_sunny:](https://tims-optimization.streamlit.app/), where you can choose values for the paramaters $a_1, a_2, b_1, b_2, c_1, c_2$ and check how the optimal solution is affected. :smile:

