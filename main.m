% DESCRIPTION:
% Top script running machine learning framework for the DC grid with the
% chosen parameters:
% - 'model': The simulink model to run representing the DC grid.
% - 'maxEpi': number of episodes, i.e. how many times the learning process
%             is repeated for the same. An improvement of the behavior is
%             expected along the episodes.
% - 'iterationTime': An episode is divided in iterations. At each
%                    iteration, the FC reference power is recalculated by
%                    the ML algorithm.
% - 'epsilon' & 'epsilon decay': Learning parameters
% - 'discount': Learning parameter setting the learning horizon time (i.e.
%               how important are the past actions).
% - 'sucessRate': Make the reference power noised if not 1.
% - 'weightX': How important we want this parameter relatively to the
%              others.
% - Folder: Crete folders to store the results from the current directory.

% The files linked to this main script are:
% - QlearningEMS.m
% - The simulink model (DC_grid_simple)
%%
clear all
close all
clc

%% Initialize the parameters for learning session:
simulation_parameters = struct(...
    'model','DC_grid_simple',...
    'maxEpi',1000,...
    'totalTime',2000,...
    'iterationTime',1.95,...
    'epsilon',3,...
    'epsilonDecay',0.997,...
    'discount',0.999,...
    'successRate',1,...
    'weightSOC',1,...
    'weightP_FC',1,...
    'weightP_batt',1,...
    'weightSteady',0,...
    'parentFolder','16_september',...
    'subFolder','test1'...
    );



%% Run Machine Learning
global inputsFromWS
QlearningEMS(simulation_parameters);

