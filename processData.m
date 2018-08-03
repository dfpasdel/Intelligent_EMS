%% DESCRIPTION:
% This script process the data collected at each episode of the
% QlearningEMS.m function.
% You can extract for example the efficiency of the FC stack, or the
% efficiency of the grid.
% Other criterions such as lifetime issues can be measured by looking at
% the reward directly (lifetime is not a state of the model). Looking into
% the reward gives qualitative assesment and allow to compare result with
% other sets of simulation.

%% STATUS
% Gives result.
% Little issue about the 10 fist seconds of the simulation is still to be
% fixed (see plots)


%% Load the data from a .mat file here:
% % data = 'Data_episode15.mat';  % Only SOC
% % data = 'Data_episode63.mat';  % Multi parameters
data = 'Data_episode64.mat';  % Multi parameters
% % data = 'Data_episode65.mat';  % Multi parameters
load(data);


%% Analyze data between 2 instants which have the same battery SOC.
SOCinit = resampledData.SOC_battery(2);
n = length(resampledData.SOC_battery);
nSamples = n;
while ~((resampledData.SOC_battery(n) >= SOCinit - 0.005) && (resampledData.SOC_battery(n) <= SOCinit + 0.005))
    n = n-1;
end

%% Average power of the load:
% Based on the continuous time series, not on the data collected at the end
% of each iteration. The sampling time is here the one given by the
% simulink solver.
idx_time_end = floor(resampledData.Load_profile.time(end)*(n/nSamples));
tsout = getdatasamples(resampledData.Load_profile, [11:idx_time_end]);
meanLoad = mean(tsout);

%% Average power of the FC:
meanPFC = mean(resampledData.P_FC(2:n));

%% Average power going into the battery:
meanPbatt_Charge = abs(mean(min(resampledData.P_Batt(2:n),0)));

%% Average power going out of the battery:
meanPbatt_Discharge = abs(mean(max(resampledData.P_Batt(2:n),0)));

%% Average efficiency of the stack:
% Take the values of efficiency that are different than 0 (0 mean the
% stack is off)
arrayEfficiency = [];
for i = 2:n
    if resampledData.Stack_efficiency(i) ~= 0
        arrayEfficiency = [arrayEfficiency resampledData.Stack_efficiency(i)];
    end
end
meanStackEfficiency = mean(arrayEfficiency)

%% Battery use
% How much power is coming from the battery to the load, and how much is 
% going directly from the FC to the load?
ratioBatt = meanPbatt_Discharge/meanLoad

%% Model efficiency
gridEfficiency = meanLoad/meanPFC
totalEfficiency = gridEfficiency*meanStackEfficiency

%% Reconstruct the efficiency
% Better to reconstruct the efficiency with more common values for the
% round trip cycle in the battery.
batt_eff = 0.97;
converter_eff = 0.98;
meanPbatt_Charge_Reconstructed = meanPbatt_Discharge/ (batt_eff * converter_eff^2);
meanPFC_Reconstructed = meanLoad - meanPbatt_Discharge + meanPbatt_Charge_Reconstructed;

gridEfficiency_Reconstructed = meanLoad/meanPFC_Reconstructed
totalEfficiency_Reconstructed = gridEfficiency_Reconstructed*meanStackEfficiency
