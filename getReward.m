function [r] = getReward(S,SOCpolynom,PFCpolynom)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% NOTE: 
% The Q-state is passed as a structure
%
% INPUTS:
% - Structure containing the current Qstate
% - Polynom defining the reward depending on the value of the Qstate (the
%   polynom is generated in a separate script and loaded once in the main 
%   script)

rSOC = polyval(SOCpolynom,S.SOC);
rPFC = max(polyval(PFCpolynom,S.P_FC),0);
if ((S.SOC >= 0.66) && (S.SOC <= 0.7) && (S.dP_Batt == 1)) ...
        || ((S.SOC >= 0.7) && (S.SOC <= 0.74) && (S.dP_Batt == -1)) % Give a bonus for good SOC
    rSOC = 2*rSOC;
end
if ((S.SOC >= 0.9) && (S.dP_Batt == -1)) || ((S.SOC <= 0.5) && (S.dP_Batt == 1))
    rSOC = 2*rSOC;
    fprintf('dPbatt penalty\n')
end
if (S.SOC >= 0.63) && (S.SOC <= 0.77)
    r = rSOC + rPFC;
else
    r = rSOC;
end
% Adding dP_batt avoid being stucked in bad condition (e.g. when SOC = 1,
% the the reduction of the recharging-power of the battery is privilegied)

end