function initialize_model_constants(model,MC)

% DESCRIPTION:
% Initialize the model constants such as nominal voltage of the battery,
% capacity of the battery, initial SOC...
%
% INPUTS: model, model constants (MC)

assignin('base','MC',MC); % Simulink is working in the base WS

load_system(model);
set_param(model,'FastRestart','off');

batteryMask = Simulink.Mask.get([model,'/Battery']); 
% Get the battery mask containing model constants
% In the Edit Mask window, the paramaters listed below have to be in
% 'run-to-run' tunable mode to apply changes each time their value is
% modified. See documentation.
%%
Nominal_voltage = batteryMask.Parameters(1,5); 
Nominal_voltage.set('Value','MC.nominal_voltage');

Rated_capacity = batteryMask.Parameters(1,6);
Rated_capacity.set('Value','MC.rated_capacity');

Initial_SOC = batteryMask.Parameters(1,7);
Initial_SOC.set('Value','MC.initial_SOC');

Battery_response_time = batteryMask.Parameters(1,8);
Battery_response_time.set('Value','MC.battery_response_time');

Checkbox = batteryMask.Parameters(1,9); 
Checkbox.set('Value','MC.nominal_voltage');

Max_capacity = batteryMask.Parameters(1,10); 
Max_capacity.set('Value','MC.max_capacity');

Cut_off_voltage = batteryMask.Parameters(1,11); 
Cut_off_voltage.set('Value','MC.cut_off_voltage');

Fully_charged_voltage = batteryMask.Parameters(1,12); 
Fully_charged_voltage.set('Value','MC.fully_charged_voltage');

Nominal_discharge_current = batteryMask.Parameters(1,13); 
Nominal_discharge_current.set('Value','MC.nominal_discharge_current');

Internal_R = batteryMask.Parameters(1,14); 
Internal_R.set('Value','MC.internal_R');

Capacity_at_nominal_voltage = batteryMask.Parameters(1,15); 
Capacity_at_nominal_voltage.set('Value','MC.capacity_at_nominal_voltage');

Exponential_zone = batteryMask.Parameters(1,16); 
Exponential_zone.set('Value','MC.exponential_zone');


end