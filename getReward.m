function [rSOC,rP_FC,rP_batt] = getReward(S,actionIdx)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% NOTE: 
% The Q-state is passed as a structure
%
% INPUTS:
% - Structure containing the current Qstate

% REWARD for the SOC:
SOCtarget = 0.7;
rSOC = 1 - abs(S.SOC-SOCtarget)/0.15;
rSOC = max(rSOC,-0.8);
if (S.SOC <= 0.5) && (actionIdx ~= 3)... % Low SOC and ot incresasing the FC power
        || (S.SOC >= 0.9) && (actionIdx ~= 2) % High SOC and not decreasing the FC power
    rSOC = rSOC - 0.2;
end

% REWARD for the FC power
if S.P_FC <= 0.7
    rP_FC = 1;
else
    rP_FC = 0;
end

% REWARD for the Battery power
rP_batt = 1.6 - 2*abs(S.P_batt);
rP_batt = min(1,rP_batt);
rP_batt = max(0,rP_batt);

end