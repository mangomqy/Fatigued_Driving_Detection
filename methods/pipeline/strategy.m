function [Gauge] = strategy(gauge, lamp_state)

%  alert state
if lamp_state == 0
    if gauge <20
        Gauge = unifrnd(8,17);
    elseif gauge < 50 && gauge >=20
        Gauge = unifrnd(18,28);
    elseif gauge < 80 && gauge >=50
        Gauge = unifrnd(29,40);
    elseif gauge < 100 && gauge >=80
        Gauge = unifrnd(40,50);
    else
        Gauge = 50;
    end
    % drowsy state
elseif lamp_state == 1
    if gauge <20
        Gauge = unifrnd(51,60);
    elseif gauge < 50 && gauge >=20
        Gauge = unifrnd(61,70);
    elseif gauge < 80 && gauge >=50
        Gauge = unifrnd(71,80);
    elseif gauge < 100 && gauge >=80
        Gauge = unifrnd(81,90);
    else
        Gauge = 50;
    end
elseif lamp_state == 2
    if gauge <20
        Gauge = unifrnd(35,44);
    elseif gauge < 50 && gauge >=20
        Gauge = unifrnd(45,54);
    elseif gauge < 80 && gauge >=50
        Gauge = unifrnd(55,64);
    elseif gauge < 100 && gauge >=80
        Gauge = unifrnd(65,74);
    else
        Gauge = 50;
    end
end

end