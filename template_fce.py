import qexpy as q
import numpy as np
import pandas as pd 
#import qexpy.plotting as plt
import matplotlib.pyplot as plt
from matplotlib import container
import locale
# Set to German locale to get comma decimal separater
#locale.setlocale(locale.LC_NUMERIC, "sk_SK.utf8")

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
        return q.Measurement(self.mean().value, np.sqrt(sum_of_squares)/len(self.errors))
    



def fitNplot(xdata,ydata,model,guess, xlabel=None, ylabel=None, ax=None ,name=None, exclude=None, fmt1='o', fmt2='-',  label=None):

    tmp = False
    if exclude:
        fit=q.fit(xdata.delete(exclude), ydata.delete(exclude) , model, parguess=guess)    
    else:
        fit=q.fit( xdata , ydata , model, parguess=guess)    

    if ax ==None:
        tmp = True
        fig, ax = plt.subplots()
    
    ax.errorbar(xdata.values,ydata.values,yerr=ydata.errors,  fmt=fmt1, label=label, capsize=4, ms=8)
    params = [exp_val.value for exp_val in fit.params]
    x_fit = np.linspace(xdata.values.min(), xdata.values.max(), 100)
    model_vals = model(x_fit,*params)
    ax.plot( x_fit, model_vals, fmt2, zorder=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

    ax.legend(handles, labels)                                  

    if name != None:
        plt.savefig(name+'.png')
        
    if tmp:
        return fit.params, ax, fig
    else:
        return fit.params, ax




def get_data(name, col=[], errors=[] , NaN = False, numpy=False, sep=";" ,decimal="," , file_type = '.csv'):
    name = str(name)+file_type 
    if file_type == '.csv':
        data = pd.read_csv(name,sep=sep ,decimal=decimal) 
    elif file_type == '.xlsx' or file_type == '.xls':
        data = pd.read_excel(name)
    
    if NaN == True:
        for i in data.columns:
            index = data[i].index[data[i].apply(np.isnan)]
            data = data.drop(index)    
        
    if len(col)>1:  
        if numpy == True:
            out = [data[c].to_numpy() for c in col]
        else:   
            out = [q.MeasurementArray(data[c].to_numpy(),errors[j]) for j, c in enumerate(col)]
    else:
        if numpy == True:
            out = data[col[0]].to_numpy()
        else:    
            out = q.MeasurementArray(data[col[0]].to_numpy(),errors[0])
        
    return out 

def first_sqn(x):
    return -int(np.floor(np.log10(abs(x))))

first_sgn = np.vectorize(first_sqn)

def to_csv(subor, veliciny, names=[]):
    df = pd.DataFrame()
    if len(names)==0:
        nazov=[str(j) for j in range(len(veliciny))]
    else:
        nazov=names
    for i, vel in enumerate(veliciny):
        df[nazov[i]] = vel
        
    df.to_csv(subor+".csv",sep=";",decimal=",",index=False)


def to_table(colomns, index=0, error = True, inline_error = False, save=False, file_name='data_to_table'):
    df = pd.DataFrame()
    
    for col in colomns:
        errors=[]
        values=[]
        
        sigma = np.any(col.errors == 0) 
        for j in range(len(col)):
            if not sigma:
                if error :
                    if first_sgn(col.errors[j]) > 0: 
                        errors.append(round(col.errors[j], first_sgn(col.errors[j])))
                        values.append(round(col.values[j],first_sgn(errors[-1])))
                    else:
                        errors.append(int(round(col.errors[j],first_sgn(col.errors[j]))))
                        values.append(int(round(col.values[j],first_sgn(errors[-1]))))

                else:
                    values.append(round(col.values[j],first_sgn(col.errors[j])))
            else: 
                values = np.round_(col.values, 1)
                break
        
        name = col.name
        
        if not sigma and inline_error:
            for k, val in enumerate(values):
                values[k] = r"${} \pm {}$".format(str(val).replace('.', ','),str(errors[k]).replace('.', ','))
            
        if hasattr(col, 'un'):
            unit = col.un
            df[r"${}$ \\ $[{}]$".format(name, unit)] = values
            if len(errors)>0:
                df[r"$\sigma_{}$ \\ $[{}]$".format("{"+str(name)+"}", unit)] = errors

        else:
            df[r"${}$".format(name)] = values
            if len(errors)>0:
                df[r"$\sigma_{}$".format("{"+str(name)+"}")] = errors


    if hasattr(colomns[index], 'un'): 
        df = df.set_index(r"${}$ \\ $[{}]$".format(colomns[index].name, colomns[index].un))
    else:
        df = df.set_index(r"${}$".format(colomns[index].name))

    if save == 'csv':
        df.to_csv(file_name+'.csv',sep=";",decimal=",",index=True)

    elif save == 'tex':
        pandas_to_latex(df, file_name)

    elif save == 'latex':
        df.to_latex(file_name+'.tex',escape=False, multirow=True, multicolumn=True)
    
    return df
    


def pandas_to_latex(df, file_name):
    outfile = open(file_name+'.tex','w')
    col_setup = '{'
    for _ in range(len(df.keys())+1):
        col_setup += 'c'
    col_setup += '}' 
    string = []
    string += [r'\begin{table}[!htb]', r'\centering', r'\caption{}', r'\label{tab:}', r'\begin{tabular}'+col_setup, r'\toprule']
    col_names = r'\begin{tabular}[c]{@{}c@{}} ' + df.index.name + r' \end{tabular}  &'
    for col in df.keys():
        col_names += r'  \begin{tabular}[c]{@{}c@{}} ' + col + r' \end{tabular}  &'
    col_names = col_names[:-1] 
    string += [col_names+r'\\', r'\midrule']

    for idx, idx_val in enumerate(df.index):
        row_string = f'{idx_val}  &'
        for col_num, col in enumerate(df.keys()):
            row_string += f'  {df.iloc[idx,col_num]}  &'
        string += [row_string[:-1] + r'\\']

    #string[-1] += r' \hline'
    string += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    for val in string:
        outfile.write(val+'\n')
    outfile.close()