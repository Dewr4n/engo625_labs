function ion = iono_corr(sow,lat,lon,az,el)
ionpar=[0.1118E-07,-0.7451E-08,-0.5961E-07, 0.1192E-06,...
        0.1167E+06,-0.2294E+06,-0.1311E+06, 0.1049E+07];%2004/1/1
    

CLIGHT      =299792458.0;

%ionospheric parameters
a0 = ionpar(1);
a1 = ionpar(2);
a2 = ionpar(3);
a3 = ionpar(4);
b0 = ionpar(5);
b1 = ionpar(6);
b2 = ionpar(7);
b3 = ionpar(8);

%elevation from 0 to 90 degrees
el = abs(el);

%conversion to semicircles
lat = lat / 180;
lon = lon / 180;
az = az / 180;
el = el / 180;

f = 1 + 16*(0.53-el)^3;

psi = (0.0137 / (el+0.11)) - 0.022;

phi = lat + psi * cos(az*pi);
phi(phi > 0.416)  =  0.416;
phi(phi < -0.416) = -0.416;

lambda = lon + ((psi*sin(az*pi)) / cos(phi*pi));

ro = phi + 0.064*cos((lambda-1.617)*pi);

t = lambda*43200 + sow;
t = mod(t,86400);


a = a0 + a1*ro + a2*ro^2 + a3*ro^3;
a(a < 0) = 0;

p = b0 + b1*ro + b2*ro^2 + b3*ro^3;
p(p < 72000) = 72000;

x = (2*pi*(t-50400))/ p;

%ionospheric delay
if abs(x) < 1.57
    ion = CLIGHT * f* (5e-9 + a* (1 - (x^2)/2 + (x^4)/24));
else
    ion = CLIGHT * f* 5e-9;
end

return