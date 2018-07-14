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
r = rSOC + rPFC;
if ((S.SOC >= 0.66) && (S.SOC <= 0.7) && (S.dP_Batt == 1)) ...
        || ((S.SOC >= 0.7) && (S.SOC <= 0.74) && (S.dP_Batt == -1)) % Give a bonus for good SOC
    r = 2*r;
end
if ((S.SOC >= 0.9) && (S.dP_Batt == -1)) || ((S.SOC <= 0.5) && (S.dP_Batt == 1))
    r = 2*r;
    fprintf('dPbatt penalty\n')
end
% Adding dP_batt avoid being stucked in bad condition (e.g. when SOC = 1,
% the the reduction of the recharging-power of the battery is privilegied

end