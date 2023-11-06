clear; 
% to run this script make sure the MS_Regress package is in the current
% folder
% the package can be found in https://github.com/msperlin/MS_Regress-Matlab

% additionally, the MEX add on should be installed

M = csvread("artificial.csv", 1,0)
addpath('m_Files')

rng(0)
dep = M(:,1);
indep{1} = M(:,2:4);
indep{2} = M(:,2:4);
k = 3;
S{1} = [1 1 0 1]
advOpt.doPlots = 0
advOpt.printOut = 0
advOpt.useMex = 1
advOpt.printIter = 0

mex -setup C++;
mex mex_MS_Filter.cpp;

spec = MS_Regress_Fit(dep, indep, k, S, advOpt);

mean(abs(sort(spec.param(1:3)) - sort([0.3,0.6,0.2]')))
spec.Coeff.nS_Param{1} - 0.333
mean(abs(sort(spec.Coeff.S_Param{1}(1,:)) - sort([1.0, -0.5, 0.12])))
mean(abs(sort(spec.Coeff.S_Param{1}(2,:)) - sort([-1.5, 0.9, 0.0])))
mean(abs(sort(diag(spec.Coeff.p)) - sort([0.8, 0.75, 0.85]')))

f = @() MS_Regress_Fit(dep, indep, k, S, advOpt);
t = timeit(f)
