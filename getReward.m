function [rSOC,rP_FC,rP_batt,rSteady] = getReward(S,actionIdx)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% This function is the Q-learning interpreter in ML words. It is of utmost 
% importance for the problem to be solved and the settings have to be
% chosen carefully
%
% NOTE: 
% The Q-state is passed as a structure
%
% INPUTS:
% Structure containing the current Qstate
% 
% OUTPUTS:
% The function outputs distinct rewards for each parameters. In order to
% weight them in the calling function, the value of the reward outputed
% here must be normalized between 0 and 1.
% 
% STATUS:
% 03/08/18: Showed good results while setting a policy with SOC, P_Batt and
% P_FC. The policy including the steady input has no good result yet.

% REWARD for the SOC:
SOCtarget = 0.7; % We aim the SOC centered on 70%
rSOC = 0.5 - abs(S.SOC-SOCtarget)/0.3;
if (S.SOC >= 0.65) && (S.SOC <= 0.75) %% && (actionIdx == 1)
    rSOC = rSOC * 2;
end      
if (S.SOC <= 0.55) && (actionIdx ~= 3)... % Low SOC and ot incresasing the FC power
        || (S.SOC >= 0.85) && (actionIdx ~= 2) % High SOC and not decreasing the FC power
    rSOC = rSOC * 2;
end
rSOC = 0.5 + 0.5*rSOC; % Rescale betwen [0;1] instead of [-1;1]

% Create a coefficient to be multiplicated with Battery and FC power
% rewards to avoid interferences when SOC is bad.
coef = 1.75 - abs(S.SOC-SOCtarget)*10;
coef = max(0,coef);
coef = min(1,coef);

% REWARD for the FC power
% We aim FC power lower than 0.8 p.u.
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

% REWARD for steady input
rSteady = (1/7) * (S.Time_steady - 2);
rSteady = max(rSteady,0);
rSteady = min(rSteady,1);
rSteady = rSteady * coef;
end