import numpy as np


def square_system(alpha, npoint, t_end):

    t = np.linspace(0.0, t_end, npoint)
    y = np.zeros_like(t)

    y[0] = 1.0
    dt = t[1] - t[0]

    for i in range(1, npoint):
        y[i] = y[i-1] + dt * (- alpha * y[i-1]**2) 

    return t, y


def tri_system(alpha, npoint, t_end):

    t = np.linspace(0.0, t_end, npoint)
    y = np.zeros_like(t)

    y[0] = 1.0
    dt = t[1] - t[0]

    for i in range(1, npoint):
        y[i] = y[i-1] + dt * (- alpha * y[i-1]**3) 

    return t, y


def primitive_combustion_system(alpha, npoint, t_end):

    t = np.linspace(0.0, t_end, npoint)
    y = np.zeros_like(t)

    y[0] = alpha
    dt = t[1] - t[0]

    for i in range(1, npoint):
        y[i] = y[i-1] + dt * y[i-1]**2 * (1.0 - y[i-1]) 

    return t, y


def square_system_carlin(alpha, npoint, t_end, trunc_order):
    import carlin.transformation as ctran
    import carlin.polynomial_ode as cpoly
    import carlin.io as cio
    import carlin.utils as cutil
    import scipy.sparse as ss
    import scipy.sparse.linalg as ssl

    ysim = polygen(QQ, 'y')
    f = [-alpha * ysim ** 2]
    P = cpoly.PolynomialODE(f, 1, 2)
    Fj = ctran.get_Fj_from_model(P.funcs(), P.dim(), P.degree())
    An = ctran.truncated_matrix(trunc_order, *Fj, input_format='Fj_matrices')

    t = np.linspace(0.0, t_end, npoint)
    sim_res = np.zeros((npoint, trunc_order))
    sim_res[0, 0] = 1.0
    dt = t[1] - t[0]
    An_csr = An.tocsr()
    An_eye_csr = ss.eye(An_csr.shape[1], format='csr')
    An_left = (An_eye_csr - dt * An_csr).tocsr()
    for i in range(1, npoint):
        x_ini = [sim_res[i - 1, 0]]
        y_ini = cutil.lift(x_ini, trunc_order)
        result = ssl.spsolve(An_left, y_ini)
        sim_res[i, :] = result

    return t, sim_res[:, 0]


def tri_system_carlin(alpha, npoint, t_end, trunc_order):
    import carlin.transformation as ctran
    import carlin.polynomial_ode as cpoly
    import carlin.io as cio
    import carlin.utils as cutil
    import scipy.sparse as ss
    import scipy.sparse.linalg as ssl

    ysim = polygen(QQ, 'y')
    f = [-alpha * ysim ** 3]
    P = cpoly.PolynomialODE(f, 1, 3)
    Fj = ctran.get_Fj_from_model(P.funcs(), P.dim(), P.degree())
    An = ctran.truncated_matrix(trunc_order, *Fj, input_format='Fj_matrices')

    t = np.linspace(0.0, t_end, npoint)
    sim_res = np.zeros((npoint, trunc_order))
    sim_res[0, 0] = 1.0
    dt = t[1] - t[0]
    An_csr = An.tocsr()
    An_eye_csr = ss.eye(An_csr.shape[1], format='csr')
    An_left = (An_eye_csr - dt * An_csr).tocsr()
    for i in range(1, npoint):
        x_ini = [sim_res[i - 1, 0]]
        y_ini = cutil.lift(x_ini, trunc_order)
        result = ssl.spsolve(An_left, y_ini)
        sim_res[i, :] = result

    return t, sim_res[:, 0]


def primitive_combustion_system_carlin(alpha, npoint, t_end, trunc_order):
    import carlin.transformation as ctran
    import carlin.polynomial_ode as cpoly
    import carlin.io as cio
    import carlin.utils as cutil
    import scipy.sparse as ss
    import scipy.sparse.linalg as ssl

    ysim = polygen(QQ, 'y')
    f = [ysim ** 2 * (1.0 - ysim)]
    P = cpoly.PolynomialODE(f, 1, 3)
    Fj = ctran.get_Fj_from_model(P.funcs(), P.dim(), P.degree())
    An = ctran.truncated_matrix(trunc_order, *Fj, input_format='Fj_matrices')

    t = np.linspace(0.0, t_end, npoint)
    sim_res = np.zeros((npoint, trunc_order))
    sim_res[0, 0] = alpha
    dt = t[1] - t[0]
    An_csr = An.tocsr()
    An_eye_csr = ss.eye(An_csr.shape[1], format='csr')
    An_left = (An_eye_csr - dt * An_csr).tocsr()
    for i in range(1, npoint):
        x_ini = [sim_res[i - 1, 0]]
        y_ini = cutil.lift(x_ini, trunc_order)
        result = ssl.spsolve(An_left, y_ini)
        sim_res[i, :] = result

    return t, sim_res[:, 0]


def plot_and_record_ref_car(ref_t, ref_y, car_t, car_y, filedir):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax1 = fig.add_subplot(211)
    ax1.set_title(filedir)
    ax1.plot(ref_t, ref_y, color='k')
    ax1.scatter(car_t, car_y, color='r', s=12)
    ax1.set_ylim(0.0, 1.05)
    
    ax2 = fig.add_subplot(212)
    ax2.plot(ref_t, car_y - ref_y)

    plt.savefig(filedir + "/plot_results.png")
    plt.close()

    npoint = ref_t.shape[0]
    record_array = np.zeros((npoint, 5))
    record_array[:, 0] = ref_t
    record_array[:, 1] = ref_y
    record_array[:, 2] = car_t
    record_array[:, 3] = car_y
    record_array[:, 4] = car_y - ref_y
    np.savetxt(
            filedir + "/time_history.csv",
            record_array,
            fmt='%.8e',
            delimiter=',',
            header="ref t, ref y, car t, car y, error"
            )


def compare_method():
    import matplotlib.pyplot as plt
    import os

    # parent_dir = "simple_combustion_sytem"
    parent_dir = "simple_square_sytem"
    alpha_t = 10.0
    # alpha_t = 2.0
    # alpha_list = [pow(10, i) for i in range(-4, -3, 1)]
    alpha_list = [1.0]

    # npoint_list = [100001, 1000001]
    npoint_list = [31, 41]
    trunc_order_list = np.arange(2, 9)

    master_ref_size = 1000001

    for alpha in alpha_list:

        alpha_dir = parent_dir + "/alpha_{0:.2e}".format(float(alpha))
        t_end = alpha_t / alpha

        master_ref_t, master_ref_y = square_system(alpha, master_ref_size, t_end)
        # master_ref_t, master_ref_y = tri_system(alpha, master_ref_size, t_end)
        # master_ref_t, master_ref_y = primitive_combustion_system(alpha, master_ref_size, t_end)

        for i, npoint in enumerate(npoint_list):

            err_by_truncation = np.zeros((len(trunc_order_list), 3))
            trunc_dir = alpha_dir + "/truncation_effect_at_N={0:d}".format(int(npoint))

            slice_interval = int((master_ref_size - 1) / (npoint - 1))

            for i, trunc_order in enumerate(trunc_order_list):

                case_name = "alpha_{0:.2e}_N_{1:d}_truncation_{2:d}".format(float(alpha), int(npoint), int(trunc_order))
                data_dir = trunc_dir + "/" + case_name
                if not os.path.isdir(data_dir): os.makedirs(data_dir)

                # ref_t, ref_y = square_system(alpha, npoint, t_end)
                ref_t = master_ref_t[::slice_interval]
                ref_y = master_ref_y[::slice_interval]
                car_t, car_y = square_system_carlin(alpha, npoint, t_end, trunc_order)
                # car_t, car_y = tri_system_carlin(alpha, npoint, t_end, trunc_order)
                # car_t, car_y = primitive_combustion_system_carlin(alpha, npoint, t_end, trunc_order)

                # Recording array
                plot_and_record_ref_car(ref_t, ref_y, car_t, car_y, data_dir)

                error = car_y - ref_y
                max_error_index = np.argmax(np.abs(error))
                err_by_truncation[i, 0] = trunc_order
                err_by_truncation[i, 1] = error[max_error_index]
                err_by_truncation[i, 2] = car_t[max_error_index]

                ref_ts = ref_t[::10]
                ref_ys = ref_y[::10]
                rec_prev_array = np.zeros((ref_ts.shape[0], 2))
                rec_prev_array[:, 0] = ref_ts
                rec_prev_array[:, 1] = ref_ys
                np.savetxt(
                        data_dir + "/compare_with_large_dt.csv",
                        rec_prev_array,
                        fmt='%12e',
                        delimiter=',',
                        header="t, y"
                        )

            np.savetxt(
                    trunc_dir + "/truncation_effect.csv",
                    err_by_truncation,
                    fmt='%.8e',
                    delimiter=',',
                    header="truncation order, maximum error, t at maximum error"
                    )

            fig = plt.figure(figsize=(12, 9), dpi=200)
            ax1 = fig.add_subplot(111)
            ax1.plot(err_by_truncation[:, 0], err_by_truncation[:, 1], marker='o')
            ax1.set_title("Truncation effect")
            plt.savefig(trunc_dir + "/truncation_effect.png")
            plt.close()


compare_method()


