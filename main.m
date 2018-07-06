% DESCRIPTION:
% Script executing everything in the right order.

%% Initialization (generate various initial states)
init_for_DC_grid

%% Set the reward strategy
makeRewardCurveSOC

%% Run Machine Learning
QlearningEMS