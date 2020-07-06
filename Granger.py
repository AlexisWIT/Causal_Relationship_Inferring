import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

filename = 'output_populations_10-5000.csv'
filename2 = 'output_pops_A.csv'
st = 0
ed = 1500

# Use module 'pandas'
def read_csv_by_Pandas(file):
    df = (pd.read_csv(file, sep=",", header=0))[st:ed]
    dfIt = df['iteration']
    dfS_1 = df['1']
    dfS_2 = df['2']
    dfS_3 = df['3']
    dfS_4 = df['4']
    #dfS_5 = df['5']

    print('\n\n1 eats 3?')
    grangercausalitytests(df[['1', '3']], maxlag=[100])
    print('\n\n3 eats 1?')
    grangercausalitytests(df[['3', '1']], maxlag=[100])
    print('\n\n2 eats 3?')
    grangercausalitytests(df[['2', '3']], maxlag=[100])
    print('\n\n3 eats 2?')
    grangercausalitytests(df[['3', '2']], maxlag=[100])
    #print('\n\n5 eats 3?')
    #grangercausalitytests(df[['5', '3']], maxlag=[50])
    #print('\n\n3 eats 5?')
    #grangercausalitytests(df[['3', '5']], maxlag=[50])
    print('\n\n1 eats 4?')
    grangercausalitytests(df[['1', '4']], maxlag=[100])
    print('\n\n4 eats 1?')
    grangercausalitytests(df[['4', '1']], maxlag=[100])
    print('\n\n2 eats 4?')
    grangercausalitytests(df[['2', '4']], maxlag=[100])
    print('\n\n4 eats 2?')
    grangercausalitytests(df[['4', '2']], maxlag=[100])

    ax = plt.gca()

    df.plot(x='iteration', y='1', kind='line', ax=ax)
    df.plot(x='iteration', y='2', kind='line', ax=ax)
    df.plot(x='iteration', y='3', kind='line', ax=ax)
    df.plot(x='iteration', y='4', kind='line', ax=ax)
    #df.plot(x='iteration', y='5', kind='line', ax=ax)
    plt.show()

def read_sample_data(filename):
    pops = np.genfromtxt(filename, delimiter=',')

    pops = pops[st:,:]

    Nsp = np.shape(pops)[1]

    colours = ['g', 'r', 'b', 'y', 'c', 'k', 'm', 'o']

    for s in range(Nsp):

        plt.plot(pops[:,s], c=colours[s])

    plt.show()


def main():
    read_csv_by_Pandas(filename)
    #read_sample_data(filename2)

# def _right_squeeze(arr, stop_dim=0):
#     """
#     Remove trailing singleton dimensions

#     Parameters
#     ----------
#     arr : ndarray
#         Input array
#     stop_dim : int
#         Dimension where checking should stop so that shape[i] is not checked
#         for i < stop_dim

#     Returns
#     -------
#     squeezed : ndarray
#         Array with all trailing singleton dimensions (0 or 1) removed.
#         Singleton dimensions for dimension < stop_dim are retained.
#     """
#     last = arr.ndim
#     for s in reversed(arr.shape):
#         if s > 1:
#             break
#         last -= 1
#     last = max(last, stop_dim)

#     return arr.reshape(arr.shape[:last])

# def bool_like(value, name, optional=False, strict=False):
#     """
#     Convert to bool or raise if not bool_like

#     Parameters
#     ----------
#     value : object
#         Value to verify
#     name : str
#         Variable name for exceptions
#     optional : bool
#         Flag indicating whether None is allowed
#     strict : bool
#         If True, then only allow bool. If False, allow types that support
#         casting to bool.

#     Returns
#     -------
#     converted : bool
#         value converted to a bool
#     """
#     if optional and value is None:
#         return value
#     extra_text = ' or None' if optional else ''
#     if strict:
#         if isinstance(value, bool):
#             return value
#         else:
#             raise TypeError('{0} must be a bool{1}'.format(name, extra_text))

#     if hasattr(value, 'squeeze') and callable(value.squeeze):
#         value = value.squeeze()
#     try:
#         return bool(value)
#     except Exception:
#         raise TypeError('{0} must be a bool (or bool-compatible)'
#                         '{1}'.format(name, extra_text))

# def int_like(value, name, optional=False, strict=False):
#     """
#     Convert to int or raise if not int_like

#     Parameters
#     ----------
#     value : object
#         Value to verify
#     name : str
#         Variable name for exceptions
#     optional : bool
#         Flag indicating whether None is allowed
#     strict : bool
#         If True, then only allow int or np.integer that are not bool. If False,
#         allow types that support integer division by 1 and conversion to int.

#     Returns
#     -------
#     converted : int
#         value converted to a int
#     """
#     if optional and value is None:
#         return None
#     is_bool_timedelta = isinstance(value, (bool, np.timedelta64))

#     if hasattr(value, 'squeeze') and callable(value.squeeze):
#         value = value.squeeze()

#     if isinstance(value, (int, np.integer)) and not is_bool_timedelta:
#         return int(value)
#     elif not strict and not is_bool_timedelta:
#         try:
#             if value == (value // 1):
#                 return int(value)
#         except Exception:
#             pass
#     extra_text = ' or None' if optional else ''
#     raise TypeError('{0} must be integer_like (int or np.integer, but not bool'
#                     ' or timedelta64){1}'.format(name, extra_text))

# def array_like(obj, name, dtype=np.double, ndim=1, maxdim=None,
#                shape=None, order='C', contiguous=False, optional=False):
#     """
#     Convert array-like to a ndarray and check conditions

#     Parameters
#     ----------
#     obj : array_like
#          An array, any object exposing the array interface, an object whose
#         __array__ method returns an array, or any (nested) sequence.
#     name : str
#         Name of the variable to use in exceptions
#     dtype : {None, numpy.dtype, str}
#         Required dtype. Default is double. If None, does not change the dtype
#         of obj (if present) or uses NumPy to automatically detect the dtype
#     ndim : {int, None}
#         Required number of dimensions of obj. If None, no check is performed.
#         If the numebr of dimensions of obj is less than ndim, additional axes
#         are inserted on the right. See examples.
#     maxdim : {int, None}
#         Maximum allowed dimension.  Use ``maxdim`` instead of ``ndim`` when
#         inputs are allowed to have ndim 1, 2, ..., or maxdim.
#     shape : {tuple[int], None}
#         Required shape obj.  If None, no check is performed. Partially
#         restricted shapes can be checked using None. See examples.
#     order : {'C', 'F'}
#         Order of the array
#     contiguous : bool
#         Ensure that the array's data is contiguous with order ``order``
#     optional : bool
#         Flag indicating whether None is allowed

#     Returns
#     -------
#     ndarray
#         The converted input.

#     Examples
#     --------
#     Convert a list or pandas series to an array
#     >>> import pandas as pd
#     >>> x = [0, 1, 2, 3]
#     >>> a = array_like(x, 'x', ndim=1)
#     >>> a.shape
#     (4,)

#     >>> a = array_like(pd.Series(x), 'x', ndim=1)
#     >>> a.shape
#     (4,)
#     >>> type(a.orig)
#     pandas.core.series.Series

#     Squeezes singleton dimensions when required
#     >>> x = np.array(x).reshape((4, 1))
#     >>> a = array_like(x, 'x', ndim=1)
#     >>> a.shape
#     (4,)

#     Right-appends when required size is larger than actual
#     >>> x = [0, 1, 2, 3]
#     >>> a = array_like(x, 'x', ndim=2)
#     >>> a.shape
#     (4, 1)

#     Check only the first and last dimension of the input
#     >>> x = np.arange(4*10*4).reshape((4, 10, 4))
#     >>> y = array_like(x, 'x', ndim=3, shape=(4, None, 4))

#     Check only the first two dimensions
#     >>> z = array_like(x, 'x', ndim=3, shape=(4, 10))

#     Raises ValueError if constraints are not satisfied
#     >>> z = array_like(x, 'x', ndim=2)
#     Traceback (most recent call last):
#      ...
#     ValueError: x is required to have ndim 2 but has ndim 3

#     >>> z = array_like(x, 'x', shape=(10, 4, 4))
#     Traceback (most recent call last):
#      ...
#     ValueError: x is required to have shape (10, 4, 4) but has shape (4, 10, 4)

#     >>> z = array_like(x, 'x', shape=(None, 4, 4))
#     Traceback (most recent call last):
#      ...
#     ValueError: x is required to have shape (*, 4, 4) but has shape (4, 10, 4)
#     """
#     if optional and obj is None:
#         return None
#     arr = np.asarray(obj, dtype=dtype, order=order)
#     if maxdim is not None:
#         if arr.ndim > maxdim:
#             msg = '{0} must have ndim <= {1}'.format(name, maxdim)
#             raise ValueError(msg)
#     elif ndim is not None:
#         if arr.ndim > ndim:
#             arr = _right_squeeze(arr, stop_dim=ndim)
#         elif arr.ndim < ndim:
#             arr = np.reshape(arr, arr.shape + (1,) * (ndim - arr.ndim))
#         if arr.ndim != ndim:
#             msg = '{0} is required to have ndim {1} but has ndim {2}'
#             raise ValueError(msg.format(name, ndim, arr.ndim))
#     if shape is not None:
#         for actual, req in zip(arr.shape, shape):
#             if req is not None and actual != req:
#                 req_shape = str(shape).replace('None, ', '*, ')
#                 msg = '{0} is required to have shape {1} but has shape {2}'
#                 raise ValueError(msg.format(name, req_shape, arr.shape))
#     if contiguous:
#         arr = np.ascontiguousarray(arr, dtype=dtype)
#     return arr

# def lagmat2ds(x, maxlag0, maxlagex=None, dropex=0, trim='forward',
#               use_pandas=False):
#     """
#     Generate lagmatrix for 2d array, columns arranged by variables.

#     Parameters
#     ----------
#     x : array_like
#         Data, 2d. Observations in rows and variables in columns.
#     maxlag0 : int
#         The first variable all lags from zero to maxlag are included.
#     maxlagex : {None, int}
#         The max lag for all other variables all lags from zero to maxlag are
#         included.
#     dropex : int
#         Exclude first dropex lags from other variables. For all variables,
#         except the first, lags from dropex to maxlagex are included.
#     trim : str
#         The trimming method to use.

#         * 'forward' : trim invalid observations in front.
#         * 'backward' : trim invalid initial observations.
#         * 'both' : trim invalid observations on both sides.
#         * 'none' : no trimming of observations.
#     use_pandas : bool
#         If true, returns a DataFrame when the input is a pandas
#         Series or DataFrame.  If false, return numpy ndarrays.

#     Returns
#     -------
#     ndarray
#         The array with lagged observations, columns ordered by variable.

#     Notes
#     -----
#     Inefficient implementation for unequal lags, implemented for convenience.
#     """
#     maxlag0 = int_like(maxlag0, 'maxlag0')
#     maxlagex = int_like(maxlagex, 'maxlagex', optional=True)
#     trim = string_like(trim, 'trim', optional=True,
#                        options=('forward', 'backward', 'both', 'none'))
#     if maxlagex is None:
#         maxlagex = maxlag0
#     maxlag = max(maxlag0, maxlagex)
#     is_pandas = _is_using_pandas(x, None)

#     if x.ndim == 1:
#         if is_pandas:
#             x = pd.DataFrame(x)
#         else:
#             x = x[:, None]
#     elif x.ndim == 0 or x.ndim > 2:
#         raise ValueError('Only supports 1 and 2-dimensional data.')

#     nobs, nvar = x.shape

#     if is_pandas and use_pandas:
#         lags = lagmat(x.iloc[:, 0], maxlag, trim=trim,
#                       original='in', use_pandas=True)
#         lagsli = [lags.iloc[:, :maxlag0 + 1]]
#         for k in range(1, nvar):
#             lags = lagmat(x.iloc[:, k], maxlag, trim=trim,
#                           original='in', use_pandas=True)
#             lagsli.append(lags.iloc[:, dropex:maxlagex + 1])
#         return pd.concat(lagsli, axis=1)
#     elif is_pandas:
#         x = np.asanyarray(x)

#     lagsli = [lagmat(x[:, 0], maxlag, trim=trim, original='in')[:, :maxlag0 + 1]]
#     for k in range(1, nvar):
#         lagsli.append(lagmat(x[:, k], maxlag, trim=trim, original='in')[:, dropex:maxlagex + 1])
#     return np.column_stack(lagsli)

# def grangercausalitytests(x, maxlag, addconst=True, verbose=True):
#     """
#     Four tests for granger non causality of 2 time series.

#     All four tests give similar results. `params_ftest` and `ssr_ftest` are
#     equivalent based on F test which is identical to lmtest:grangertest in R.

#     Parameters
#     ----------
#     x : array_like
#         The data for test whether the time series in the second column Granger
#         causes the time series in the first column. Missing values are not
#         supported.
#     maxlag : {int, Iterable[int]}
#         If an integer, computes the test for all lags up to maxlag. If an
#         iterable, computes the tests only for the lags in maxlag.
#     addconst : bool
#         Include a constant in the model.
#     verbose : bool
#         Print results.

#     Returns
#     -------
#     dict
#         All test results, dictionary keys are the number of lags. For each
#         lag the values are a tuple, with the first element a dictionary with
#         test statistic, pvalues, degrees of freedom, the second element are
#         the OLS estimation results for the restricted model, the unrestricted
#         model and the restriction (contrast) matrix for the parameter f_test.

#     Notes
#     -----
#     TODO: convert to class and attach results properly

#     The Null hypothesis for grangercausalitytests is that the time series in
#     the second column, x2, does NOT Granger cause the time series in the first
#     column, x1. Grange causality means that past values of x2 have a
#     statistically significant effect on the current value of x1, taking past
#     values of x1 into account as regressors. We reject the null hypothesis
#     that x2 does not Granger cause x1 if the pvalues are below a desired size
#     of the test.

#     The null hypothesis for all four test is that the coefficients
#     corresponding to past values of the second time series are zero.

#     'params_ftest', 'ssr_ftest' are based on F distribution

#     'ssr_chi2test', 'lrtest' are based on chi-square distribution

#     References
#     ----------
#     .. [1] https://en.wikipedia.org/wiki/Granger_causality

#     .. [2] Greene: Econometric Analysis

#     Examples
#     --------
#     >>> import statsmodels.api as sm
#     >>> from statsmodels.tsa.stattools import grangercausalitytests
#     >>> import numpy as np
#     >>> data = sm.datasets.macrodata.load_pandas()
#     >>> data = data.data[['realgdp', 'realcons']].pct_change().dropna()

#     # All lags up to 4
#     >>> gc_res = grangercausalitytests(data, 4)

#     # Only lag 4
#     >>> gc_res = grangercausalitytests(data, [4])
#     """
#     x = array_like(x, 'x', ndim=2)
#     if not np.isfinite(x).all():
#         raise ValueError('x contains NaN or inf values.')
#     addconst = bool_like(addconst, 'addconst')
#     verbose = bool_like(verbose, 'verbose')
#     try:
#         lags = np.array([int(lag) for lag in maxlag])
#         maxlag = lags.max()
#         if lags.min() <= 0 or lags.size == 0:
#             raise ValueError('maxlag must be a non-empty list containing only '
#                              'positive integers')
#     except Exception:
#         maxlag = int_like(maxlag, 'maxlag')
#         if maxlag <= 0:
#             raise ValueError('maxlag must a a positive integer')
#         lags = np.arange(1, maxlag + 1)

#     if x.shape[0] <= 3 * maxlag + int(addconst):
#         raise ValueError("Insufficient observations. Maximum allowable "
#                          "lag is {0}".format(int((x.shape[0] - int(addconst)) /
#                                                  3) - 1))

#     resli = {}

#     for mlg in lags:
#         result = {}
#         if verbose:
#             print('\nGranger Causality')
#             print('number of lags (no zero)', mlg)
#         mxlg = mlg

#         # create lagmat of both time series
#         dta = lagmat2ds(x, mxlg, trim='both', dropex=1)

#         # add constant
#         if addconst:
#             dtaown = add_constant(dta[:, 1:(mxlg + 1)], prepend=False)
#             dtajoint = add_constant(dta[:, 1:], prepend=False)
#         else:
#             raise NotImplementedError('Not Implemented')
#             # dtaown = dta[:, 1:mxlg]
#             # dtajoint = dta[:, 1:]

#         # Run ols on both models without and with lags of second variable
#         res2down = OLS(dta[:, 0], dtaown).fit()
#         res2djoint = OLS(dta[:, 0], dtajoint).fit()

#         # print results
#         # for ssr based tests see:
#         # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
#         # the other tests are made-up

#         # Granger Causality test using ssr (F statistic)
#         fgc1 = ((res2down.ssr - res2djoint.ssr) /
#                 res2djoint.ssr / mxlg * res2djoint.df_resid)
#         if verbose:
#             print('ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
#                   ' df_num=%d' % (fgc1,
#                                   stats.f.sf(fgc1, mxlg,
#                                              res2djoint.df_resid),
#                                   res2djoint.df_resid, mxlg))
#         result['ssr_ftest'] = (fgc1,
#                                stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
#                                res2djoint.df_resid, mxlg)

#         # Granger Causality test using ssr (ch2 statistic)
#         fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
#         if verbose:
#             print('ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, '
#                   'df=%d' % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg))
#         result['ssr_chi2test'] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

#         # likelihood ratio test pvalue:
#         lr = -2 * (res2down.llf - res2djoint.llf)
#         if verbose:
#             print('likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d' %
#                   (lr, stats.chi2.sf(lr, mxlg), mxlg))
#         result['lrtest'] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

#         # F test that all lag coefficients of exog are zero
#         rconstr = np.column_stack((np.zeros((mxlg, mxlg)),
#                                    np.eye(mxlg, mxlg),
#                                    np.zeros((mxlg, 1))))
#         ftres = res2djoint.f_test(rconstr)
#         if verbose:
#             print('parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,'
#                   ' df_num=%d' % (ftres.fvalue, ftres.pvalue, ftres.df_denom,
#                                   ftres.df_num))
#         result['params_ftest'] = (np.squeeze(ftres.fvalue)[()],
#                                   np.squeeze(ftres.pvalue)[()],
#                                   ftres.df_denom, ftres.df_num)

#         resli[mxlg] = (result, [res2down, res2djoint, rconstr])

#     return resli

if __name__ == "__main__":
    main()