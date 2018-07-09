function [r] = getReward(S,SOCpolynom)
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

r = polyval(SOCpolynom,S.SOC);
if (S.SOC >= 0.9) && (S.dP_Batt == -1)
    r = 2*r;
    fprintf('dPbatt penalty\n')
end

end