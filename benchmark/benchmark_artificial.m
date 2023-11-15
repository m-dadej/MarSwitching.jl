clear; 
% to run this script make sure the MS_Regress package is in the current
% folder
% the package can be downloaded from github repo: https://github.com/msperlin/MS_Regress-Matlab
% after unziping the folder set current directory to m_Files and copy and
% paste there a "artificial.csv" file

% additionally, the MEX add on should be installed. You may do it from
% Home->Add-ons->get Add-ons 
% it's called "MATLAB Support for MinGW-w64 C/C++ Compiler"

M = csvread("artificial.csv", 1,0)
addpath('m_Files')

rng(01)
dep = M(:,1);
indep{1} = M(:,2:4);
indep{2} = M(:,2:4);
k = 3;
S{1} = [1 1 0 1]
advOpt.doPlots = 0
advOpt.printOut = 0
advOpt.useMex = 1
advOpt.printIter = 0
advOpt.optimizer = 'fmincon' % that's the only optimizer that finds true parameters

mex -setup C++;
mex mex_MS_Filter.cpp;

spec = MS_Regress_Fit(dep, indep, k, S, advOpt);

mean(abs(sort(sqrt(spec.param(1:3))) - sort([0.3,0.6,0.2]')))
spec.Coeff.nS_Param{1} - 0.333
mean(abs(sort(spec.Coeff.S_Param{1}(1,:)) - sort([1.0, -0.5, 0.12])))
mean(abs(sort(spec.Coeff.S_Param{1}(2,:)) - sort([-1.5, 0.9, 0.0])))
mean(abs(sort(diag(spec.Coeff.p)) - sort([0.8, 0.75, 0.85]')))

f = @() MS_Regress_Fit(dep, indep, k, S, advOpt); % handle to function
timeit(f)
