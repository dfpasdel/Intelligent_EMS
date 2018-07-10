% DESCRIPTION:
% Script executing everything in the right order.

%% Initialization (generate various initial states)
init_for_DC_grid

%%
clear all
close all
clc

%% Set the reward strategy
makeRewardCurveSOC

%% Initialize the parameters for learning session:
simulation_parameters_1 = struct(...
    'model','DC_grid_V2',...
    'maxEpi',2,...
    'totalTime',20,...
    'iterationTime',1.3,...
    'learnRate',0.99,...
    'epsilon',0.5,...
    'epsilonDecay',0.9996,...
    'discount',0.9,...
    'successRate',1,...
    'rewardCurveSOC','rewardCurveSOC.mat',...
    'subFolder','simulation1'...
    );

simulation_parameters_2 = struct(...
    'model','DC_grid_V2',...
    'maxEpi',2,...
    'totalTime',20,...
    'iterationTime',1.3,...
    'learnRate',0.99,...
    'epsilon',0.5,...
    'epsilonDecay',0.9996,...
    'discount',0.7,...
    'successRate',1,...
    'rewardCurveSOC','rewardCurveSOC.mat',...
    'subFolder','simulation2'...
    );

simulation_parameters_3 = struct(...
    'model','DC_grid_V2',...
    'maxEpi',2,...
    'totalTime',20,...
    'iterationTime',1.3,...
    'learnRate',0.99,...
    'epsilon',0.5,...
    'epsilonDecay',0.9996,...
    'discount',0.5,...
    'successRate',1,...
    'rewardCurveSOC','rewardCurveSOC.mat',...
    'subFolder','simulation3'...
    );

simulation_parameters_4 = struct(...
    'model','DC_grid_V2',...
    'maxEpi',2,...
    'totalTime',20,...
    'iterationTime',1.3,...
    'learnRate',0.99,...
    'epsilon',0.5,...
    'epsilonDecay',0.9996,...
    'discount',0.95,...
    'successRate',1,...
    'rewardCurveSOC','rewardCurveSOC.mat',...
    'subFolder','simulation4'...
    );

arrayParam = [...
    simulation_parameters_1...
    simulation_parameters_2...
    simulation_parameters_3...
    simulation_parameters_4...
    ];

%% Run Machine Learning in series

directory = '10_july_series';
tic
QlearningEMS(arrayParam(1),directory);
QlearningEMS(arrayParam(2),directory);
QlearningEMS(arrayParam(3),directory);
QlearningEMS(arrayParam(4),directory);
tseries = toc;

%% Run Machine Learning in parallel

directory = '10_july_parallel';
parpool('local',4)
tic
parfor i = 1:length(arrayParam)
    QlearningEMS(arrayParam(i),directory);
end
tparallel = toc;

%%