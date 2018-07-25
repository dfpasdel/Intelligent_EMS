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
rSOC = 0.5 + 0.5*rSOC; % Rescale betwen [0;1] instead of [-1;1]

% Create a coefficient to be multiplicated with Battery and FC power
% rewards to avoid interferences when SOC is bad.
coef = 1.75 - abs(S.SOC-SOCtarget)*10eward;
coef = max(0,coef);
coef = min(1,coef);

% REWARD for the FC power
if S.P_FC <= 0.8
    rP_FC = 1;
else
    rP_FC = 0;
end
rP_FC = rP_FC * coef;

% REWARD for the Battery power
rP_batt = 1.6 - 2*abs(S.P_batt);
rP_batt = min(1,rP_batt);
rP_batt = max(0,rP_batt);
rP_batt = rP_batt * coef;

end