function [rSOC] = getReward(S,actionIdx)
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
end