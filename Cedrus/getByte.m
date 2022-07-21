
function byte = getByte(val, index)
    byte = bitand(bitshift(val,-8*(index-1)), 255);
end