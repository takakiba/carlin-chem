import cantera as ct
ct.suppress_thermo_warnings()

import carlin.transformation as ctran
import carlin.polynomial_ode as cpoly
import carlin.io as cio
import carlin.utils as cutil

import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import scipy.linalg as spl

import signal

# mechanism = "./sandiego/chem.yaml"
mechanism = "./sandiego/chem.yaml"
temperature = 2000.0
pressure = 101325.0
# fuel = 'CH4'
fuel = 'H2'
oxidizer = 'O2:0.21, N2:0.79'
equivalence_ratio = 0.80
# computational_time = 2.0e-4
computational_time = 4.0e-5
order_of_system = 3
truncation_order = 3
# matrix_update_interval = 2.0e-8
matrix_update_interval = 2.0e-10


class CarlemannReaction:
    def __init__(
            self,
            mechanism,
            temperature,
            pressure,
            fuel,
            oxidizer,
            equivalence_ratio,
            computational_time,
            order_of_system,
            truncation_order,
            matrix_update_interval
            ):
        import os

        self.mechanism = mechanism
        self.temperature = temperature
        self.pressure = pressure
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.equivalence_ratio = equivalence_ratio
        self.computational_time = computational_time
        self.order_of_system = order_of_system
        self.truncation_order = truncation_order
        self.matrix_update_interval = matrix_update_interval

        self.data_dir_prefix = "./carlin_reac"
        self.data_dir_ver = "/ver002"
        self.data_dir_case = "/fuel_{0}/"\
                "temperature_{1:.0f}_presssure_{2:.0f}_equivalence_ratio_{3:.2f}_time_{4:.3e}/truncation_{5:d}/"\
                "matrix_update_interval_{6:.3e}".format(
                                     self.fuel,
                                     self.temperature,
                                     self.pressure,
                                     self.equivalence_ratio,
                                     self.computational_time,
                                     self.truncation_order,
                                     self.matrix_update_interval
                                     )
        self.data_dir_summary = self.data_dir_prefix + self.data_dir_ver + self.data_dir_case
        self.data_dir_detail = self.data_dir_summary + "/detail"

        if not os.path.isdir(self.data_dir_summary): os.makedirs(self.data_dir_summary)
        if not os.path.isdir(self.data_dir_detail): os.makedirs(self.data_dir_detail)

        self.gas = self.extract_mech()
        self.gas.TP = self.temperature, self.pressure
        self.gas.set_equivalence_ratio(self.equivalence_ratio, self.fuel, self.oxidizer)
        self.states = ct.SolutionArray(self.gas, extra='t')

        self.t_raw = []
        self.y_raw = []
        self.omega_raw = []

    
    def extract_mech(self):
        import warnings

        all_species = ct.Species.listFromFile(self.mechanism)
        fil_species = []

        for S in all_species:
            comp = S.composition
            if 'C' in comp:
                if self.fuel == 'H2':
                    continue
                elif self.fuel == 'CH4' and comp['C'] >1:
                    continue
            if 'N' in comp and comp['N'] != 2:
                continue
            if 'Ar' in comp:
                continue
            if 'He' in comp:
                continue
            if S == 'H2O2':
                continue
            fil_species.append(S)

        species_names = {S.name for S in fil_species}

        ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species)
        all_reactions = ct.Reaction.listFromFile(self.mechanism, ref_phase)
        fil_reactions = []

        for R in all_reactions:
            if not all(reactant in species_names for reactant in R.reactants):
                continue
            if not all(product in species_names for product in R.products):
                continue
            fil_reactions.append(R)

        gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=fil_species, reactions=fil_reactions)

        return gas


    def calc_reference(self):
        gas_ref = self.extract_mech()
        gas_ref.TP = self.temperature, self.pressure
        gas_ref.set_equivalence_ratio(self.equivalence_ratio, self.fuel, self.oxidizer)
        r = ct.IdealGasConstPressureReactor(gas_ref, energy='off')
        sim = ct.ReactorNet([r])

        states = ct.SolutionArray(gas_ref, extra=['t'])
        states.append(gas_ref.state, t=0.0)

        num_data = 1000
        dt_max = self.computational_time / float(num_data)
        while sim.time < self.computational_time:
            sim.advance(sim.time + dt_max)
            states.append(gas_ref.state, t=sim.time)

        states.write_csv(self.data_dir_detail + "/reference_data.csv")
        return states


    def reaction_data_for_carlin(self, return_x=False):
        R = self.gas.reactions()
        f_rate = self.gas.forward_rate_constants
        r_rate = self.gas.reverse_rate_constants
        n_spec = self.gas.species_names
        c_spec = self.gas.concentrations

        x = polygens(QQ, ["x{}".format(i) for i in range(self.gas.n_species)])
        f = []

        # with open(self.data_dir_detail + "/construct_chemical_data.txt", 'w') as fw:
        for i, n in enumerate(n_spec):
            # fw.write("Species: {0}: {1}\n".format(i, n))
            net_production_rate = 0.0 * x[0]
            for j, rxn in enumerate(R):
                reac_sp = 1.0
                if (n in rxn.reactants) or (n in rxn.products):
                    reac_sp =  f_rate[j]
                    if n in rxn.reactants:
                        reac_sp = -1.0 * reac_sp
                    # fw.write("\t # based on reaction no.{0} : {1}\n".format(j, rxn))
                    # fw.write("\t\t{0:+.2e} ".format(reac_sp))
                    for sp in rxn.reactants:
                        # reac_sp = reac_sp * x[self.gas.species_index(sp)] ** rxn.reactants[sp]
                        reac_sp = reac_sp * (
                                self.gas.density / 
                                self.gas.molecular_weights[self.gas.species_index(sp)]
                                ) ** rxn.reactants[sp] * \
                                x[self.gas.species_index(sp)] ** rxn.reactants[sp] 
                        # if rxn.reactants[sp] == 1.0:
                        #     fw.write("x {0}({1}) ".format(sp, self.gas.species_index(sp)))
                        # else:
                        #     fw.write("x ({0}({1}) ** {2}) ".format(sp, self.gas.species_index(sp), rxn.reactants[sp]))
                    net_production_rate = net_production_rate + reac_sp

                    reac_sp = r_rate[j]
                    if n in rxn.products:
                        reac_sp = -1.0 * reac_sp
                    # fw.write("{0:+.2e} ".format(reac_sp))
                    for sp in rxn.products:
                        # reac_sp = reac_sp * x[self.gas.species_index(sp)] ** rxn.products[sp]
                        reac_sp = reac_sp * (
                                self.gas.density /
                                self.gas.molecular_weights[self.gas.species_index(sp)]
                                ) ** rxn.products[sp] * \
                                x[self.gas.species_index(sp)] ** rxn.products[sp] 
                        # if rxn.products[sp] == 1.0:
                        #     fw.write("x {0}({1}) ".format(sp, self.gas.species_index(sp)))
                        # else:
                        #     fw.write("x ({0}({1}) ** {2}) ".format(sp, self.gas.species_index(sp), rxn.products[sp]))
                    # fw.write("\n")
                    net_production_rate = net_production_rate + reac_sp
            net_production_rate = self.gas.molecular_weights[i] * net_production_rate / self.gas.density
            f.append(net_production_rate)
        if return_x:
            return f, x
        else:
            return f


    def prepare_An(self):
        f = self.reaction_data_for_carlin()
        P = cpoly.PolynomialODE(f, self.gas.n_species, self.order_of_system)
        Fj = ctran.get_Fj_from_model(P.funcs(), P.dim(), P.degree())
        An = ctran.truncated_matrix(self.truncation_order, *Fj, 
                                    input_format='Fj_matrices')
        return An


    def simple_ODE(self, t_start=0.0, t_end=-1):
        import numpy as np
        if t_end < 0:
            t_end = self.computational_time

        ref_result = self.calc_reference()

        y_0 = vector(QQ, self.gas.Y.tolist())
        y_1 = y_0
        t_0 = t_start
        t_1 = t_0 + self.matrix_update_interval

        # t_raw = []
        # y_raw = []
        self.states.append(self.gas.state, t=t_0)

        self.t_raw.append(t_0)
        self.y_raw.append(y_0)

        mat_i = identity_matrix(QQ, n=self.gas.n_species)

        output_interval = 10
        iter_num = 0

        while t_1 < t_end:
            print("\rCalc. {0:.6e} to {1:.6e}".format(t_0, t_1), end='')
            f, x = self.reaction_data_for_carlin(return_x=True)
            j = jacobian(f, x)
            js = j.subs(dict(zip(x, self.gas.Y.tolist())))

            Jac = js

            Mat_r = mat_i - (t_1 - t_0) * Jac
            y_1 = Mat_r.solve_right(y_0)

            self.t_raw.append(t_1)
            self.y_raw.append(y_1)
            self.states.append(self.gas.state, t=t_1)

            self.gas.Y = y_1
            y_0 = y_1

            t_0 = t_1
            t_1 += self.matrix_update_interval

            if iter_num % output_interval == 0:
                self.write_csvs("sODE_temporary")
                self.plot_results(ref_result, file_prefix="sODE_temporary")

            iter_num += 1
        print(" ... Done!")


    def save_time_elapsed(self, t_inv, t_prp, prefix="comparison"):
        import numpy as np
        with open(self.data_dir_detail + "/time_elapsed_{}.csv".format(prefix), 'w') as ft:
            ft.write("{0}, {1}, {2}\n".format("i", "time matrix prepare", "time solve linear algebra"))
            ft.write("Average\n")
            ft.write(", {0:.6e}, {1:.6e}\n".format(np.mean(t_prp), np.mean(t_inv)))
            for i, t in enumerate(t_prp):
                ft.write("{0:8d}, {1:.6e}, {2:.6e}\n".format(i, t_prp[i], t_inv[i]))


    def carlin_ODE(self, t_start=0.0, t_end=-1):
        import numpy as np
        import time

        if t_end < 0:
            t_end = self.computational_time
        tlist_mat_prepare = []
        tlist_mat_inverse = []

        ref_result = self.calc_reference()

        y_0 = self.gas.Y
        y_1 = self.gas.Y
        t_0 = t_start
        t_1 = t_0 + self.matrix_update_interval

        self.t_raw.append(t_0)
        self.y_raw.append(y_0)

        self.states.append(self.gas.state, t=t_0)
        An = self.prepare_An().tocsr()

        An_eye_csr = ss.eye(An.shape[1], format='csr')

        record_interval = 10000
        iter_num = 0

        while t_1 < t_end:
            print("\rCalc. {0:.6e} to {1:.6e}".format(t_0, t_1), end='')
            x_0 = cutil.lift(y_0.tolist(), self.truncation_order)
            reaction_rate_base = An * x_0
            self.omega_raw.append(reaction_rate_base[:self.gas.n_species])

            An_left = (An_eye_csr - 0.50 * (t_1 - t_0) * An)
            An_right = (An_eye_csr + 0.50 * (t_1 - t_0) * An)

            t0_inv = time.time()
            x_1 = ssl.spsolve(
                    An_left, 
                    An_right * x_0
                    )
            t1_inv = time.time() - t0_inv
            tlist_mat_inverse.append(t1_inv)

            y_1 = np.array(x_1[:self.gas.n_species])

            self.t_raw.append(t_1)
            self.y_raw.append(y_1)

            self.gas.TP = self.temperature, self.pressure
            self.gas.Y = y_1
            self.states.append(self.gas.state, t=t_1)
            y_0 = y_1

            t0_prp = time.time()
            An = self.prepare_An().tocsr()
            t1_prp = time.time() - t0_prp
            tlist_mat_prepare.append(t1_prp)

            t_0 = t_1
            t_1 += self.matrix_update_interval

            if iter_num % record_interval == 0:
                self.plot_results(ref_result, file_prefix="temporary")
                self.write_csvs(file_prefix="temporary")
                self.save_time_elapsed(tlist_mat_inverse, tlist_mat_prepare, prefix="temporary")

            iter_num += 1
        self.save_time_elapsed(tlist_mat_inverse, tlist_mat_prepare)


    def plot_data_from_cantera_solution(self, ct_solution):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 9), dpi=200)
        ax1 = fig.add_subplot(111)
        for i, nsp in enumerate(ct_solution.species_names):
            if i < 10:
                ax1.plot(ct_solution.t, ct_solution.Y[:, i], label='{}'.format(nsp), linestyle='-')
            elif i < 20:
                ax1.plot(ct_solution.t, ct_solution.Y[:, i], label='{}'.format(nsp), linestyle='--')
            elif i < 30:
                ax1.plot(ct_solution.t, ct_solution.Y[:, i], label='{}'.format(nsp), linestyle=':')
        ax1.set_xlabel("Time [sec]")
        ax1.set_ylabel("Mass Fraction [-]")
        ax1.legend()

        plt.tight_layout()
        plt.savefig(self.data_dir_detail + "/reference_data.png")
        plt.close()


    def plot_results(self, ct_solution, file_prefix="comparison"):
        import numpy as np
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 9), dpi=200)
        ax1 = fig.add_subplot(111)
        for i, nsp in enumerate(ct_solution.species_names):
            if i < 10:
                ax1.plot(ct_solution.t, ct_solution.Y[:, i], label='{}'.format(nsp), linestyle='-')
                ax1.scatter(self.states.t, self.states.Y[:, i], label='{}'.format(nsp), marker='o')
            elif i < 20:
                ax1.plot(ct_solution.t, ct_solution.Y[:, i], label='{}'.format(nsp), linestyle='--')
                ax1.scatter(self.states.t, self.states.Y[:, i], label='{}'.format(nsp), marker='x')
            elif i < 30:
                ax1.plot(ct_solution.t, ct_solution.Y[:, i], label='{}'.format(nsp), linestyle=':')
                ax1.scatter(self.states.t, self.states.Y[:, i], label='{}'.format(nsp), marker='^')
        ax1.set_xlabel("Time [sec]")
        ax1.set_ylabel("Mass Fraction [-]")

        self.states.Y[self.states.Y == np.inf] = np.nan
        self.states.Y[self.states.Y == -np.inf] = np.nan
        ymax = np.nanmax(self.states.Y)

        ax1.set_xlim(0.0, self.states.t[-1])
        ax1.legend(bbox_to_anchor=(0, 1.01, 1.0, 0.40), loc='lower left', ncol=9)

        ax1.set_ylim(0.0, 1.05 * ymax)
        plt.tight_layout()
        plt.savefig(self.data_dir_detail + "/results_{}_data.png".format(file_prefix))

        ax1.set_ylim(0.0, 0.105 * ymax)
        plt.tight_layout()
        plt.savefig(self.data_dir_detail + "/results_{}_data_x10.png".format(file_prefix))

        ax1.set_ylim(0.0, 0.0105 * ymax)
        plt.tight_layout()
        plt.savefig(self.data_dir_detail + "/results_{}_data_x100.png".format(file_prefix))

        plt.close()
        

    def eval_error(self):
        import numpy as np
        ref_result = self.calc_reference()

        eval_interval = 5.0e-6
        te0 = 0.0
        te1 = eval_interval

        while te1 < self.computational_time:
            self.carlin_ODE(t_start=te0, t_end=te1)
            
            self.plot_results(ref_result, file_prefix="EvalError")
            self.gas.Y = ref_result.Y[np.argmin(np.abs(ref_result.t - te1)), :]
            te0 = te1
            te1 += eval_interval

        print("Writeing results ...")
        self.write_csvs()
        print("Done !")


    def write_csvs(self, file_prefix='comparison'):
        import numpy as np
        '''
        np.savetxt(
                self.data_dir_detail + "/carlemann_result_time_{}.csv".format(file_prefix),
                self.states.t, 
                delimiter=',', 
                fmt='%.6e',
                header='Time [sec]'
                )
        np.savetxt(
                self.data_dir_detail + "/carlemann_result_mass_{}.csv".format(file_prefix),
                self.states.Y, 
                delimiter=',', 
                fmt='%.6e',
                header=','.join(self.gas.species_names)
                )
        '''
        self.states.write_csv(self.data_dir_detail + "/carlemann_result_{}.csv".format(file_prefix))
        np.savetxt(
                self.data_dir_detail + "/carlemann_result_raw_time_{}.csv".format(file_prefix),
                np.array(self.t_raw), 
                delimiter=',', 
                fmt='%.6e',
                header='Time [sec]'
                )
        np.savetxt(
                self.data_dir_detail + "/carlemann_result_raw_mass_{}.csv".format(file_prefix),
                np.array(self.y_raw), 
                delimiter=',', 
                fmt='%.6e',
                header=','.join(self.gas.species_names)
                )
        np.savetxt(
                self.data_dir_detail + "/carlemann_result_raw_reactionRate_{}.csv".format(file_prefix),
                np.array(self.omega_raw), 
                delimiter=',', 
                fmt='%.6e',
                header=','.join(self.gas.species_names)
                )
        

    def debug(self):
        import time

        print("Preparing reference data ...", end='')
        ref_result = self.calc_reference()
        print("Done !")

        print("Recording reference data ...", end='')
        self.plot_data_from_cantera_solution(ref_result)
        print("Done !")

        t_0 = time.time()
        print("Starting calclation ...")
        # An = self.prepare_An()
        self.carlin_ODE()
        print("Done !")
        t_1 = time.time() - t_0
        with open(self.data_dir_detail + "/total_time_taken.txt", "w") as ft:
            ft.write("time elapsed : {0:.8e}".format(t_1))
        print("recoded time elapsed ({0:.4e})".format(t_1))

        print("Writing results ...")
        # An = self.prepare_An()
        self.plot_results(ref_result)
        self.write_csvs()
        print("Done !")

        
    def handler(self, signal, frame):
        import sys
        ct_solution = self.calc_reference()
        self.plot_results(ct_solution, file_prefix="temporary")
        self.write_csvs(file_prefix='temporary')
        sys.exit(0)


cr = CarlemannReaction(
        mechanism,
        temperature,
        pressure,
        fuel,
        oxidizer,
        equivalence_ratio,
        computational_time,
        order_of_system,
        truncation_order,
        matrix_update_interval
        )
signal.signal(signal.SIGINT, cr.handler)
# cr.eval_error()
cr.debug()
# cr.simple_ODE()


