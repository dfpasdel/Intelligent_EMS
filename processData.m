%%
clear all
close all
clc

%%

% % data = 'Data_episode15.mat';  % Only SOC
% % data = 'Data_episode63.mat';  % Multi parameters
data = 'Data_episode64.mat';  % Multi parameters
% % data = 'Data_episode65.mat';  % Multi parameters
load(data);
% %%
% % Analyze data between 2 instants which have the same battery SOC.
% SOCinit = systemStatesTab.SOC_battery(2);
% n = length(systemStatesTab.SOC_battery);
% while ~((systemStatesTab.SOC_battery(n) >= SOCinit - 0.005) && (systemStatesTab.SOC_battery(n) <= SOCinit + 0.005))
%     n = n-1;
% end
% 
% % Average power of the load:
% meanLoad = mean(systemStatesTab.Load_profile(2:n));
% 
% % Average power of the FC:
% meanPFC = mean(systemStatesTab.P_FC(2:n));
% 
% % Average power going into the battery:
% meanPbatt_Charge = abs(mean(min(systemStatesTab.P_Batt(2:n),0)));
% 
% % Average power going out of the battery:
% meanPbatt_Discharge = abs(mean(max(systemStatesTab.P_Batt(2:n),0)));
% 
% % Average efficiency of the stack:
% % Take the values of efficiency that are different than 0 (0 mean the
% % stack is off)
% arrayEfficiency = [];
% for i = 2:n
%     if systemStatesTab.Stack_efficiency(i) ~= 0
%         arrayEfficiency = [arrayEfficiency systemStatesTab.Stack_efficiency(i)];
%     end
% end
% meanStackEfficiency = mean(arrayEfficiency);


% Analyze data between 2 instants which have the same battery SOC.
SOCinit = resampledData.SOC_battery(2);
n = length(resampledData.SOC_battery);
nSamples = n;
while ~((resampledData.SOC_battery(n) >= SOCinit - 0.005) && (resampledData.SOC_battery(n) <= SOCinit + 0.005))
    n = n-1;
end

% Average power of the load:
idx_time_end = floor(resampledData.Load_profile.time(end)*(n/nSamples));
tsout = getdatasamples(resampledData.Load_profile, [11:idx_time_end]);

meanLoad = mean(tsout);

% Average power of the FC:
meanPFC = mean(resampledData.P_FC(2:n));

% Average power going into the battery:
meanPbatt_Charge = abs(mean(min(resampledData.P_Batt(2:n),0)));

% Average power going out of the battery:
meanPbatt_Discharge = abs(mean(max(resampledData.P_Batt(2:n),0)));

% Average efficiency of the stack:
% Take the values of efficiency that are different than 0 (0 mean the
% stack is off)
arrayEfficiency = [];
for i = 2:n
    if resampledData.Stack_efficiency(i) ~= 0
        arrayEfficiency = [arrayEfficiency resampledData.Stack_efficiency(i)];
    end
end
meanStackEfficiency = mean(arrayEfficiency)

%%   
ratioBatt = meanPbatt_Discharge/meanLoad
meanPbatt_Charge_R = meanPbatt_Discharge/ (0.97*0.96*0.96)
meanPFC_R = meanLoad - meanPbatt_Discharge + meanPbatt_Charge_R;
gridEfficiency = meanLoad/meanPFC;
gridEfficiency_R = meanLoad/meanPFC_R
totalEfficiency = gridEfficiency*meanStackEfficiency;
totalEfficiency_R = gridEfficiency_R*meanStackEfficiency
