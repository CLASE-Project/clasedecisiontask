% ---Function 'Cedrus_TTL'---
% 
% For use with Cedrus c-pod to transmit hexidecmial TTL pulses
% Requirements: 'get Byte.m' and 'setPulseDiration.m' functions on path 
%
% Marielle L. Darwin | John A. Thompson
% July 14 2022 | Last update: July 19 2022

function [ports, device] = Cedrus_TTL()
clear device
device_found = 0;
ports = serialportlist("available");

for p = 1:length(ports)
    device = serialport(ports(p),115200,"Timeout",1);
    device.flush()
    write(device,"_c1","char")
    query_return = read(device,5,"char");
    if length(query_return) > 0 && query_return == "_xid0"
        device_found = 1;
        %return
    end
end

if device_found == 0
    error("No XID device found. Exiting.")
    %return
end

setPulseDuration(device, 1);

disp("Raised all output lines for 1 millisecond.")
end