""" 
A bunch of utility functions - some are not used in the final version of the scripts

"""


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.style.use('seaborn-whitegrid')


def damped_pendulum(y, t, α, β):
    θ, dθdt = y     
    return dθdt, -α*dθdt - β*np.sin(θ)


def generate_damped_pendulum_solution(N_SAMPLES=25000, α=0.2, β=8.91, Δ=0.1, 
    x0=[-1.193, -3.876], dataset=False, t_end=25, noise=0.0):
    """
        :param N_SAMPLES - size of dataset if dataset=True
        :param α - parameter governing differential equation
        :param β - parameter governing differential equation
        :param Δ - time lag between solution state pairs
        :param x0 - initial conditions of differential equation, used if dataset=False
        :param dataset 
            True -  generate dataset of solution states pairs,  
            False - generate a single solution for a specified set of initial 
                    conditions x0.
        :param t_end: value of last timestep of ode solver, used if dataset=False
        :param noise - amount of random noise to add to solution states, 
                       used if dataset=True
    """
    if not dataset:
        t = np.linspace(0, t_end, 100000)
        sol = odeint(damped_pendulum, x0, t, args=(α, β))
        x = sol[:,0]
        dxdt = sol[:,1]
        return t, x, dxdt
    else:
        # Define time array for ODE solver to integrate over. 
        t = np.linspace(0, Δ, 10000)
        
        # Arrays that will store Z(1) and Z(2)
        X = np.zeros((N_SAMPLES, 2))
        Y = np.zeros((N_SAMPLES, 2))

        for i in range(0,N_SAMPLES):
            if i%100==0:
                print("\r generating {} / {}".format(i+100, N_SAMPLES), end='')
            # Generate random initial conditions
            θ_0 = np.random.uniform(-np.pi, np.pi)
            dθdt_0 = np.random.uniform(-2*np.pi, 2*np.pi)
            x0 = [θ_0, dθdt_0]
            # Generate solution from t=0 to t=Δ
            sol = odeint(damped_pendulum, x0, t, args=(α, β))
            # Generate noise terms (no noise if `noise=0`)
            ε_1 = np.random.uniform(-noise, noise)
            ε_2 = np.random.uniform(-noise, noise)
            # Extract solution state at t=Δ
            θ_Δ = sol[-1,0]
            dθdt_Δ = sol[-1,1]

            X[i,0] = θ_0 + ε_1
            X[i,1] = dθdt_0 + ε_1

            Y[i,0] = θ_Δ + ε_2
            Y[i,1] = dθdt_Δ + ε_2

        return X, Y


def generate_batches(x: np.array, y: np.array, batch_size: int):
    for i in range(0, x.shape[1], batch_size):
        yield (
            x.take(indices=range(i, min(i + batch_size, x.shape[1])), axis=1),
            y.take(indices=range(i, min(i + batch_size, y.shape[1])), axis=1)
        )
        
        
def plot_history(model): 
    train_loss = model.train_cost_history
    valid_loss = model.valid_cost_history
    epochs = range(len(train_loss))
    plt.figure(figsize=(16,7))
    plt.plot(epochs, train_loss, 'b', alpha=0.7, label="Training")
    plt.plot(epochs, valid_loss, 'r', alpha=0.7, label="Validation")
    plt.title('Training and Validation Loss',fontsize=22)
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=20)
    plt.legend(prop={"size":20})
    plt.show()
    

def plot_keras_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(figsize=(16,7))
    plt.plot(epochs, loss, 'b', label="Training")
    plt.plot(epochs, val_loss, 'r', label="Validation")
    plt.title('Training and Validation Loss',fontsize=22)
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=20)
    plt.legend(prop={"size":20})
    plt.show()
    
    
def predict_solution(model, params_values, Δ=0.1, t_end=20):

    t_steps = np.arange(0,t_end, Δ)
    X_pred = np.zeros((len(t_steps), 2))

    x_0 = np.array([-1.193, -3.876])
    x_Δ = np.array([-1.193, -3.876])
    
    for i in range(0, len(t_steps)):
        x_0 = x_Δ
        x_Δ = model.predict(x_0, params_values)
        x_Δ = np.squeeze(x_Δ.T)
        X_pred[i] = x_Δ   
        
    t_steps = t_steps + Δ
    return t_steps, X_pred


def predict_keras_solution(model, Δ=0.1, t_end=20):

    t_steps = np.arange(0,t_end, Δ)
    X_pred = np.zeros((len(t_steps), 2))

    x_0 = np.array([-1.193, -3.876])
    x_Δ = x_0
    
    for i in range(0, len(t_steps)):
        x_0 = x_Δ
        x_Δ = model.predict(np.expand_dims(x_0, axis=0))
        x_Δ = np.squeeze(x_Δ)
        X_pred[i] = x_Δ   
        
    t_steps = t_steps + Δ
    return t_steps, X_pred


def predict_multiple_solutions(n_solutions, model, params_values, Δ=0.1, t_end=20, α=0.2, β=8.91):
    
    t = np.linspace(0, t_end, 10000)
    t_steps = np.arange(0,t_end, Δ)
    
    X_pred = np.zeros((n_solutions, len(t_steps)))
    X_true = np.zeros((n_solutions, len(t)))
    
    for i in range(0, n_solutions):
        
        pred = np.zeros((len(t_steps),))

        θ_0 = np.random.uniform(-np.pi, np.pi)
        dθdt_0 = np.random.uniform(-2*np.pi, 2*np.pi)
        
        x_0 = [θ_0, dθdt_0]
        x_Δ = x_0
        
        sol = odeint(damped_pendulum, x_0, t, args=(α, β))
        
        for j in range(0, len(t_steps)):
            x_0 = x_Δ
            x_Δ = model.predict(x_0, params_values)
            x_Δ = np.squeeze(x_Δ.T)
            pred[j] = x_Δ[0]   
                
        X_true[i] = sol[:,0]
        X_pred[i] = pred
    
    t_steps = t_steps + Δ
    return t, t_steps, X_true, X_pred


def predict_multiple_solutions_keras(n_solutions, model, Δ=0.1, t_end=20, α=0.2, β=8.91):
    
    t = np.linspace(0, t_end, 10000)
    t_steps = np.arange(0,t_end, Δ)
    
    X_pred = np.zeros((n_solutions, len(t_steps)))
    X_true = np.zeros((n_solutions, len(t)))
    
    X0_arr = np.zeros((n_solutions, 2))
    XΔ_arr = np.zeros((n_solutions, 2))
    
    for i in range(0, n_solutions):        
        x0_i = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-2*np.pi, 2*np.pi)]
        
        sol = odeint(damped_pendulum, x0_i, t, args=(α, β))
        
        X0_arr[i] = x0_i
        X_true[i] = sol[:,0]
        
    XΔ_arr = X0_arr
    
    for j in range(0, len(t_steps)):
        print("\r Stepping {} / {}".format(j+1, len(t_steps)), end='')
        X0_arr = XΔ_arr
        XΔ_arr = model.predict(X0_arr)
        X_pred[:,j] = XΔ_arr[:,0]
        
        
    t_steps = t_steps + Δ
    return t, t_steps, X_true, X_pred


def get_error(t_actual, x_actual, t_pred, x_pred, t_end):
    Nt = t_actual.shape[0]
    errors = []
    for i in range(0,len(t_pred)):
        t_idx = int((Nt/t_end)*t_pred[i] - 1)
        x_true = x_actual[t_idx]
        x_est = x_pred[i]
        err = np.abs((x_true - x_est)/x_true)*100
        #print("x true = {:.4f}  x est = {:.4f}  error = {:.4f}".format(x_true, x_est, err))
        errors.append(err)
    return errors


def get_errors(t, t_steps, X_true, X_pred, t_end):
    Nt = t.shape[0]
    X_errors = np.zeros((len(X_true), len(t_steps)))
    for i in range(0,len(t_steps)):
        t_idx = int((Nt/t_end)*t_steps[i] - 1)
        err = np.abs((X_true[:,t_idx] - X_pred[:,i])/X_true[:,t_idx])*100
        #print("x true = {:.4f}  x est = {:.4f}  error = {:.4f}".format(x_true, x_est, err))
        X_errors[:,i] = err
    return X_errors


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_trajectory(t, t_steps, x, dxdt, X_pred, title):
    
    plt.figure(figsize=(16,8))
    
    plt.gcf().suptitle(title, fontsize=20, y=1.0)
    
    plt.subplot(121)
    plt.title(r"$x$ and $dx/dt$", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.plot(t, x, "-", c="r", linewidth=3, label=r"$x$ Reference")
    plt.plot(t, dxdt, "-", c="orange", linewidth=3, label=r"$dx/dt$ Reference")
    plt.plot(t_steps, X_pred[:,0], ".", c="b", markersize=10, label=r"$x$ Approximation")
    plt.plot(t_steps, X_pred[:,1],".", c="m", markersize=10, label=r"$dx/dt$ Approximation")
    plt.legend(prop={"size":12}, loc="upper right", frameon=True, markerscale=1)
    
    plt.subplot(122)
    plt.title("Phase Space", fontsize=18)
    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$dx/dt$", fontsize=18)
    plt.plot(x, dxdt, "-", c="r", linewidth=2, label="Reference")
    plt.plot(X_pred[:,0], X_pred[:,1], ":.", c="b", markersize=10, label="Approximation")
    plt.legend(prop={"size":12}, loc="upper left", frameon=True, markerscale=1)
    
    plt.tight_layout()
    plt.show()

    
def plot_multiple_trajectories(t, t_steps, X_true_arr, X_pred_arr, n_samples, title, up_ylim=[-25,25]):
    
    fig = plt.figure(figsize=(16,16))
    
    gs = gridspec.GridSpec(2, 2)
    
    plt.gcf().suptitle(title, fontsize=20, y=1.0)


    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])


    ax1.set_title("Predicted Trajectories", fontsize=18)
    ax1.set_xlabel("t", fontsize=18)
    ax1.set_ylabel("$ x(t) $", fontsize=18)
    for i in range(0,n_samples):
        ax1.plot(t_steps, X_pred_arr[i], "-", linewidth=2, markersize=4)
    ax1.set_ylim(up_ylim)


    ax2.set_title("True Trajectories", fontsize=18)
    ax2.set_xlabel("t", fontsize=18)
    ax2.set_ylabel("$ x(t) $", fontsize=18)
    for i in range(0,n_samples):
        ax2.plot(t, X_true_arr[i], "-", linewidth=2)
    ax2.set_ylim(up_ylim)


    X_errs = get_errors(t, t_steps, X_true_arr, X_pred_arr, t_end=t_steps[-1])
    X_errs_avg = np.mean(X_errs, axis=0)
    ax3.set_title(r"Average Percent Error", fontsize=18)
    ax3.set_xlabel("t", fontsize=18)
    ax3.set_ylabel("$ \% Error $", fontsize=18)
    ax3.plot(t_steps, X_errs_avg, ":.", linewidth=2, c="r", markerfacecolor='blue', markersize=6)
    ax3.set_yscale("log")


    plt.tight_layout()
    plt.show()

    
    
def get_true_vector_field(n_solutions, Δ=0.1, t_end=5, α=0.2, β=8.91):
    
    t_steps = np.arange(0,t_end, Δ)
    t = np.linspace(0, t_end, 10000)
    Nt = t.shape[0]
    
    
    X0_arr = np.zeros((int(n_solutions * (len(t_steps)-1) + 1), 2))
    XΔ_arr = np.zeros((int(n_solutions * (len(t_steps)-1) + 1), 2))
    
    # Generate full solutions
    for i in range(0, n_solutions):        
        x0_i = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-2*np.pi, 2*np.pi)]
        sol = odeint(damped_pendulum, x0_i, t, args=(α, β))
        
        x0_coords = np.zeros((len(t_steps)-1, 2))
        xΔ_coords = np.zeros((len(t_steps)-1, 2))
        
        for j in range(1, len(t_steps)):
            t0_idx = int((Nt/t_end)*t_steps[j-1])
            tΔ_idx = int((Nt/t_end)*t_steps[j])
            # print("[t0_idx, tΔ_idx] = [{}, {}]".format(t0_idx, tΔ_idx))
            
            x0 = sol[t0_idx]
            xΔ = sol[tΔ_idx]
            
            x0_coords[j-1] = x0
            xΔ_coords[j-1] = xΔ
            
        idx_begin = int(i*(len(t_steps)-1))
        idx_end = int((i+1)*(len(t_steps)-1))
        
        X0_arr[idx_begin:idx_end] = x0_coords
        XΔ_arr[idx_begin:idx_end] = xΔ_coords
        
        # print("[idx_begin, idx_end] = [{}, {}]".format(idx_begin, idx_end))
        # print()
        
    #print(len(X0_arr))
        
    return X0_arr, XΔ_arr
    
    
    
def get_predicted_vector_field(n_solutions, model, params_values, Δ=0.1, t_end=5, α=0.2, β=8.91):
    
    t_steps = np.arange(0,t_end, Δ)
    t = np.linspace(0, t_end, 10000)
    Nt = t.shape[0]
    
    X0_arr = np.zeros((int(n_solutions * (len(t_steps)-1) + 1), 2))
    XΔ_arr = np.zeros((int(n_solutions * (len(t_steps)-1) + 1), 2))
    
    for i in range(0, n_solutions):        
        x0 = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-2*np.pi, 2*np.pi)]
        xΔ = x0
        
        sol = odeint(damped_pendulum, x0, t, args=(α, β))
        
        x0_coords = np.zeros((len(t_steps)-1, 2))
        xΔ_coords = np.zeros((len(t_steps)-1, 2))
        
        for j in range(1, len(t_steps)):
            x0 = xΔ
            xΔ = model.predict(x0, params_values)
            xΔ = np.squeeze(xΔ.T)
            
            x0_coords[j-1] = x0
            xΔ_coords[j-1] = xΔ
            
        idx_begin = int(i*(len(t_steps)-1))
        idx_end = int((i+1)*(len(t_steps)-1))
        
        X0_arr[idx_begin:idx_end] = x0_coords
        XΔ_arr[idx_begin:idx_end] = xΔ_coords
                
    return X0_arr, XΔ_arr



def get_normalized_vector_coords(X0_arr, XΔ_arr, scale=0.02):
    x = X0_arr[:-1,0]
    y = X0_arr[:-1,1]

    u = XΔ_arr[:-1,0] - x 
    v = XΔ_arr[:-1,1] - y  
    
    # Normalize the arrows:
    u_norm = u / np.sqrt(u**2 + v**2);
    v_norm = v / np.sqrt(u**2 + v**2);

    # Scale them a bit
    u_norm = u_norm * scale * np.sqrt(u**2 + v**2)
    v_norm = v_norm * scale * np.sqrt(u**2 + v**2)

    c = y
    return x, y, u_norm, v_norm, c