import qexpy as q
import numpy as np
import pandas as pd
# import qexpy.plotting as plt
import matplotlib.pyplot as plt
from matplotlib import container
import re
# import locale
# Set to German locale to get comma decimal separater
# locale.setlocale(locale.LC_NUMERIC, "sk_SK.utf8")

plt.rcdefaults()

# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


class MA(q.MeasurementArray):

    def set_un(self, unit):
        self.un = unit 

    def add_MA(self, MeasurementArrays):
        data = self.values
        err = self.errors
        for measArr in MeasurementArrays:
            data = np.append(data, measArr.values)
            err = np.append(err, measArr.errors)
        return MA(data, err)

    def avg(self):
        sum_of_squares = sum([error**2 for error in self.errors])
        return q.Measurement(self.mean().value, np.sqrt(sum_of_squares) / len(self.errors))


def plot(xdata, ydata, xlabel=None, ylabel=None, ax=None, fname=None, fmt='o', label=None):

    own_fig = False
    if ax is None:
        own_fig = True
        fig, self_ax = plt.subplots()
    else:
        self_ax = ax

    self_ax.errorbar(xdata.values, ydata.values, yerr=ydata.errors, fmt=fmt, label=label, capsize=4, ms=8)
    
    if xlabel is not None:
        self_ax.set_xlabel(xlabel)
    elif self_ax.xaxis.get_label().get_text() == '':
        if hasattr(xdata, 'un') and hasattr(xdata, 'name'):
            self_ax.set_xlabel(f'${xdata.name} \ [{unit_to_latex(xdata.un, plt=True)}]$')
            
        elif hasattr(xdata, 'name'):
            self_ax.set_xlabel(f'${xdata.name} \ [1]$')

    if ylabel is not None:
        self_ax.set_ylabel(ylabel)
    elif self_ax.yaxis.get_label().get_text() == '': 
        if hasattr(ydata, 'un'):
            self_ax.set_ylabel(f'${ydata.name} \ [{unit_to_latex(ydata.un, plt= True)}]$')
        else:
            self_ax.set_ylabel(f'${ydata.name} \ [1]$')
    
    handles, labels = self_ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    self_ax.legend(handles, labels)                            

    if fname is not None:
        plt.savefig(fname + '.png')

    if own_fig:
        return self_ax, fig
    else:
        return self_ax


def fitNplot(xdata, ydata, model, guess, xlabel=None, ylabel=None, ax=None, fname=None, exclude=None, fmt1='o', model_fmt='-', label=None):

    if exclude:
        fit = q.fit(xdata.delete(exclude), ydata.delete(exclude), model, parguess=guess)    
    else:
        fit = q.fit(xdata, ydata, model, parguess=guess)    

    own_fig = False
    if ax is None:
        own_fig = True
        fig, self_ax = plt.subplots()
    else:
        self_ax = ax

    params = [exp_val.value for exp_val in fit.params]
    x_fit = np.linspace(xdata.values.min(), xdata.values.max(), 10 * len(xdata.values))
    model_vals = model(x_fit, *params)
    self_ax.plot(x_fit, model_vals, model_fmt, zorder=10)

    self_ax = plot(xdata, ydata, xlabel, ylabel, self_ax, fname, fmt1, label)

    if own_fig:
        return fit.params, self_ax, fig
    else:
        return fit.params, self_ax


def get_data(name, col=[], errors=[], NaN=False, numpy=False, sep=";", decimal=",", file_type='.csv'):
    name = str(name) + file_type
    if file_type == '.csv':
        data = pd.read_csv(name, sep=sep, decimal=decimal)
    elif file_type == '.xlsx' or file_type == '.xls':
        data = pd.read_excel(name)

    if NaN:
        for i in data.columns:
            index = data[i].index[data[i].apply(np.isnan)]
            data = data.drop(index)

    if len(col) > 1:
        if numpy:
            out = [data[c].to_numpy() for c in col]
        else:
            out = [q.MeasurementArray(data[c].to_numpy(), errors[j]) for j, c in enumerate(col)]
    else:
        if numpy:
            out = data[col[0]].to_numpy()
        else:
            out = q.MeasurementArray(data[col[0]].to_numpy(), errors[0])

    return out


def first_sqn(x):
    return -int(np.floor(np.log10(abs(x))))


first_sgn = np.vectorize(first_sqn)


def to_table(colomns, index=0, error=True, inline_error=False, save=False, fname='data_to_table'):
    df = pd.DataFrame()
    
    for col in colomns:
        errors = []
        values = []
        
        sigma = np.any(col.errors == 0)
        for j in range(len(col)):
            if not sigma:
                if error:
                    if first_sgn(col.errors[j]) > 0: 
                        errors.append(round(col.errors[j], first_sgn(col.errors[j])))
                        values.append(round(col.values[j], first_sgn(errors[-1])))
                    else:
                        errors.append(int(round(col.errors[j], first_sgn(col.errors[j]))))
                        values.append(int(round(col.values[j], first_sgn(errors[-1]))))

                else:
                    if first_sgn(col.errors[j]) > 0: 
                        values.append(round(col.values[j], first_sgn(col.errors[j])))
                    else:
                        values.append(int(round(col.values[j], first_sgn(col.errors[j]))))
            else:
                values = np.round_(col.values, 1)
                break

        name = col.name
        if not sigma and inline_error:
            for k, val in enumerate(values):
                values[k] = r"${} \pm {}$".format(str(val).replace('.', ','), str(errors[k]).replace('.', ','))

        if hasattr(col, 'un'):
            unit = unit_to_latex(col.un)
            df[r"${}$ \\ $[{}]$".format(name, unit)] = values
            if len(errors) > 0:
                df[r"$\sigma_{}$ \\ $[{}]$".format("{" + str(name) + "}", unit)] = errors

        else:
            df[r"${}$".format(name)] = values
            if len(errors) > 0:
                df[r"$\sigma_{}$".format("{" + str(name) + "}")] = errors

    if type(index) is int:
        if hasattr(colomns[index], 'un'): 
            df = df.set_index(r"${}$ \\ $[{}]$".format(colomns[index].name, unit_to_latex(colomns[index].un)))
        else:
            df = df.set_index(r"${}$".format(colomns[index].name))
    elif isinstance(index, pd.Series):
        df = df.set_index(index)

    if save == 'csv':
        df.to_csv(fname + '.csv', sep=";", decimal=",", index=True)

    elif save == 'tex':
        if type(index) is int:  
            pandas_to_latex(df, fname)
        else:
            pandas_to_latex(df, fname, True)

    elif save == 'latex':
        df.to_latex(fname + '.tex', escape=False, multirow=True, multicolumn=True)
    
    return df
    

def unit_to_latex(string, plt=None):
    if plt is None:
        return re.sub(r"([a-zA-Z]+)", r'\\text{\1}', string)
    else:
        return re.sub(r"([a-zA-Z]+)", r'\\mathrm{\1}', string)


def pandas_to_latex(df, fname, index=False):
    outfile = open(fname + '.tex', 'w')
    col_setup = '{'
    for _ in range(len(df.keys()) + 1):
        col_setup += 'c'
    col_setup += '}'
    string = []
    string += [r'\begin{table}[!htb]', r'\centering', r'\caption{}', r'\label{tab:}', r'\begin{tabular}' + col_setup, r'\toprule']
    col_names = r'\begin{tabular}[c]{@{}c@{}} ' + df.index.name + r' \end{tabular}  &'
    for col in df.keys():
        col_names += r'  \begin{tabular}[c]{@{}c@{}} ' + col + r' \end{tabular}  &'
    col_names = col_names[:-1] 
    string += [col_names + r'\\', r'\midrule']

    old_idx = None
    old_multirow_num = 0
    new_multirow_num = 0
    for idx, idx_val in enumerate(df.index):
        if index:
            multi_bool = False
            if idx_val == old_idx:
                row_string = r'  &'
                new_multirow_num += 1
            else:
                old_idx = idx_val
                row_string = r'\multirow{...}{*}{' + f'{idx_val}' + r'} &'
                new_multirow_num = 1

            if new_multirow_num <= old_multirow_num:
                multi_bool = True
                string[-old_multirow_num] = string[-old_multirow_num].replace('...', f'{old_multirow_num}')
            multirow_num = new_multirow_num

            if multi_bool and idx != len(df.index): 
                string[-1] = string[-1] + r' \hline'
            old_multirow_num = multirow_num

        else:
            row_string = f'{idx_val}  &'

        for col_num, col in enumerate(df.keys()):
            row_string += f'  {df.iloc[idx,col_num]}  &'
        string += [row_string[:-1] + r'\\']

    if index:
        string[-old_multirow_num] = string[-old_multirow_num].replace('...', f'{old_multirow_num}')

    string += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    for val in string:
        outfile.write(val + '\n')
    outfile.close()

